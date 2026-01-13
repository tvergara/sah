import json
import os
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names

from sah.algorithms.strategies.base_strategy import BaseStrategy

from .log_utils import get_bw
from .quantization import LoraQuantLinear, LoraSVDQuantLinear, quantize_apply_lora_model
from .straight_through import BayesianBitsQuantizer

# No torch.compile usage; run in eager mode


def prepare_model(model, learn_gates, quantize, learn_scales):
    trainable_params = ["lora"]
    excluded_params = []

    quantizable = [w for w in model.modules() if hasattr(w, "create_quantizer")]
    for module in quantizable:
        module.quantizer = module.create_quantizer()

    # Initialize gamma_2 for LoraSVDQuantLinear layers to ensure they are included in the optimizer
    for name, module in model.named_modules():
        if isinstance(module, LoraSVDQuantLinear):
            # Ensure the quantizer is created (it should be by the loop above, but safe check)
            if module.lora_E_quantizer.quantizer is None:
                module.lora_E_quantizer.quantizer = (
                    module.lora_E_quantizer.create_quantizer()
                )

            bbq = module.lora_E_quantizer.quantizer
            # If prune_rank is enabled and gamma_2 is not yet initialized
            if bbq.prune_rank and bbq.gamma_2 is None:
                # lora_E has shape (r, 1), so size(0) is r.
                # We initialize gamma_2 with size r-1, matching the logic in straight_through.py
                r = module.r
                gamma_2 = torch.Tensor((r - 1) * [bbq.gamma_2_init])
                bbq.gamma_2 = torch.nn.Parameter(gamma_2.to(bbq.device))

    if learn_gates:
        if quantize:
            trainable_params.extend([f"gamma_{x}" for x in [2, 4, 8, 16, 32]])
        else:
            trainable_params.append("gamma_2")
            excluded_params.extend([f"gamma_{x}" for x in [4, 8, 16, 32]])

    if learn_scales:
        trainable_params.extend(["x_min", "x_max", "s_2"])

    for name, param in model.named_parameters():
        # if name.startswith("model"):
        param.requires_grad = False
        for trainable_param in trainable_params:
            exclude = False
            for excl in excluded_params:
                if excl in name:
                    exclude = True
                    break

            if trainable_param in name and not exclude:
                param.requires_grad = True
                break
        # else:
        #     param.requires_grad = False
    # if args.train_head:
    #     model.lm_head.weight.requires_grad = True
    #     if model.lm_head.bias is not None:
    #         model.lm_head.bias.requires_grad = True

    return model


def get_accelerate_model(
    model,
    quantize,
    prune_rank,
    fixed_full_precision,
    fixed_8bit,
    lora_r=64,
    lora_module="q_proj,k_proj,v_proj,up_proj,down_proj,o_proj,gate_proj",
):
    if quantize or prune_rank:
        quant_params = {
            "learned_scale": "scale",
            "gamma_4_init": 6.0,
            "gamma_8_init": 6.0,
            "gamma_16_init": 6.0,
            "gamma_32_init": 6.0,
            "fixed_full_precision": fixed_full_precision,
            "fixed_8bit": fixed_8bit,
        }
        model = quantize_apply_lora_model(
            model,
            "",
            prune_rank=prune_rank,
            method="bayesian_bits",
            n_bits=16,
            quantize_lora_layers_only=True,
            lora_rank=lora_r,
            lora_weights=tuple(lora_module.split(",")),
            **quant_params,
        )

    return model


class BLoRAStrategy(BaseStrategy):
    def __init__(
        self,
        r=64,
        quant_loss_lambda: float = 1e-3,
        rank_loss_lambda: float = 1e-3,
        gates_lr: float = 0.01,
        scales_lr: float = 0.01,
        prune_rank: bool = True,
        quantize: bool = True,
        learn_gates: bool = True,
        learn_scales: bool = True,
        fixed_full_precision: bool = False,
        fixed_8bit: bool = False,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        self.r = r
        self.quant_loss_lambda = quant_loss_lambda
        self.rank_loss_lambda = rank_loss_lambda
        self.gates_lr = gates_lr
        self.scales_lr = scales_lr
        self.prune_rank = prune_rank
        self.quantize = quantize
        self.learn_gates = learn_gates
        self.learn_scales = learn_scales
        self.fixed_full_precision = fixed_full_precision
        self.fixed_8bit = fixed_8bit
        self.weight_decay = weight_decay

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)
        model = get_accelerate_model(
            pl_module.model,
            quantize=self.quantize,
            prune_rank=self.prune_rank,
            fixed_full_precision=self.fixed_full_precision,
            fixed_8bit=self.fixed_8bit,
            lora_r=self.r,
        )
        pl_module.model = prepare_model(
            model, self.learn_gates, self.quantize, self.learn_scales
        )
        pl_module.model = pl_module.model.to(pl_module.device)

        for module in pl_module.model.modules():
            try:
                if hasattr(module, "device"):
                    module.device = pl_module.device
            except:  # noqa: E722
                pass

    def on_validation_epoch_end(self, pl_module):
        self.bits, bits_per_layer = self.compute_bits(pl_module)
        pl_module.log("val/bits", self.bits, prog_bar=True)

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        performance = pl_module.trainer.logged_metrics.get("val/performance", None)

        if performance is not None:
            experiment_id = pl_module.trainer.logger.experiment.id
            eval_run_id = str(uuid.uuid4())
            experiment_name = pl_module.experiment_name
            result_file = pl_module.result_file
            dataset_name = pl_module.dataset_name
            model_name = pl_module.model_name

            result = {
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "eval_run_id": eval_run_id,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "performance": (
                    performance.item() if hasattr(performance, "item") else performance
                ),
                "bits": self.bits,
                "seed": pl_module.hparams.seed,
                "strategy_hparams": vars(pl_module.hparams.strategy),
            }

            bits_path = Path(os.environ.get("SCRATCH", "./outputs")) / "blora_bits"
            bits_path.mkdir(parents=True, exist_ok=True)
            bits_file = bits_path / f"{experiment_id}_{eval_run_id}_bits_per_layer.json"
            with open(bits_file, "w") as bf:
                json.dump(bits_per_layer, bf, indent=4)

            with open(result_file, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

            if hasattr(self.dataset_handler, "save_generations"):
                self.dataset_handler.save_generations(eval_run_id)

    def validation_step(self, pl_module, batch, batch_idx):
        return super().validation_step(pl_module, batch, batch_idx)

    def configure_optimizers(self, pl_module):
        opt_model = pl_module.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [
            name
            for name in decay_parameters
            if "bias" not in name
            and "gamma" not in name
            and "x_min" not in name
            and "x_max" not in name
            and "s_2" not in name
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad and "bias" in n)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad and "gamma" in n)
                ],
                "weight_decay": self.weight_decay,
                "lr": self.gates_lr,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n not in decay_parameters
                        and p.requires_grad
                        and ("x_min" in n or "x_max" in n or "s_2" in n)
                    )
                ],
                "weight_decay": self.weight_decay,
                "lr": self.scales_lr,
            },
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters)

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss

        gate_loss_quant, gate_loss_rank = calculate_gate_loss(pl_module.model)
        gate_loss = (
            gate_loss_quant * self.quant_loss_lambda
            + gate_loss_rank * self.rank_loss_lambda
        )
        pl_module.log(
            "train_gate_loss_quant",
            gate_loss_quant,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        pl_module.log(
            "train_gate_loss_rank",
            gate_loss_rank,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        pl_module.log(
            "train_gate_loss", gate_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        pl_module.log(
            "train_base_loss", loss, on_step=True, on_epoch=False, prog_bar=True
        )
        return loss + gate_loss

    def compute_bits(self, pl_module):
        total_bits = 0
        bits_per_layer = {}
        for name, module in pl_module.named_modules():
            if isinstance(module, LoraSVDQuantLinear):
                if module.prune_rank:
                    q = module.lora_E_quantizer.quantizer
                    was_training = q.training
                    q.eval()
                    gates = q.get_gates()
                    gate_2 = gates[0]
                    q.train(was_training)

                    if gate_2 is not None:
                        r_eff = torch.count_nonzero(gate_2).item() + 1
                    else:
                        r_eff = module.r
                else:
                    r_eff = module.r

                bw_A = get_bw(module.lora_A_quantizer)
                bw_B = get_bw(module.lora_B_quantizer)
                bw_E = get_bw(module.lora_E_quantizer)

                bits_A = module.out_features * r_eff * bw_A
                bits_B = module.in_features * r_eff * bw_B
                bits_E = r_eff * bw_E
                layer_bits = bits_A + bits_B + bits_E

                total_bits += layer_bits
                bits_per_layer[name] = layer_bits

            elif isinstance(module, LoraQuantLinear):
                r = module.r
                bw_A = get_bw(module.lora_A_quantizer)
                bw_B = get_bw(module.lora_B_quantizer)

                bits_A = module.out_features * r * bw_A
                bits_B = module.in_features * r * bw_B
                layer_bits = bits_A + bits_B

                total_bits += layer_bits
                bits_per_layer[name] = layer_bits

        return total_bits, bits_per_layer


def calculate_gate_loss(model):
    regularizer_quant, regularizer_rank = 0.0, 0.0
    for name, module in model.named_modules():
        if isinstance(module, BayesianBitsQuantizer):
            regularizer = module.regularizer()
            regularizer_quant += regularizer[0]
            regularizer_rank += regularizer[1]

    return regularizer_quant, regularizer_rank
