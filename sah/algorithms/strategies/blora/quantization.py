import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import lora_layers
from .autoquant_utils import (
    QuantLinear,
    get_module_args,
    non_bn_module_map,
    quantize_sequential,
)
from .straight_through import QuantizationHijacker, Quantizer


def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)


class LoraQuantizationHijacker(QuantizationHijacker):
    def __init__(
        self,
        *args,
        method="bayesian_bits",
        n_bits=16,
        act_momentum=0.1,
        percentile=None,
        fix_range_on_first_forward=False,
        scale_domain="linear",
        gating_method="l0",
        gamma_4_init=2.0,
        gamma_8_init=2.0,
        gamma_16_init=2.0,
        gamma_32_init=2.0,
        learned_scale=False,
        clip_input=False,
        checkpointing=False,
        include_pruning=False,
        reg_type="const",
        prune_only=False,
        fixed_8bit=False,
        fixed_48bit=False,
        fixed_full_precision=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__(
            *args,
            method=method,
            n_bits=n_bits,
            percentile=percentile,
            act_momentum=act_momentum,
            fix_range_on_first_forward=fix_range_on_first_forward,
            scale_domain=scale_domain,
            gating_method=gating_method,
            gamma_4_init=gamma_4_init,
            gamma_8_init=gamma_8_init,
            gamma_16_init=gamma_16_init,
            gamma_32_init=gamma_32_init,
            learned_scale=learned_scale,
            clip_input=clip_input,
            checkpointing=checkpointing,
            include_pruning=include_pruning,
            reg_type=reg_type,
            prune_only=prune_only,
            fixed_8bit=fixed_8bit,
            fixed_48bit=fixed_48bit,
            fixed_full_precision=fixed_full_precision,
            device=device,
            **kwargs,
        )

        self.lora_A_quantizer = Quantizer(
            method,
            self.N,
            False,
            device=device,
            percentile=percentile,
            fix_range_on_first_forward=fix_range_on_first_forward,
            scale_domain=scale_domain,
            gating_method=gating_method,
            gamma_4_init=gamma_4_init,
            gamma_8_init=gamma_8_init,
            gamma_16_init=gamma_16_init,
            gamma_32_init=gamma_32_init,
            learned_scale=learned_scale,
            clip_input=clip_input,
            checkpointing=checkpointing,
            include_pruning=include_pruning,
            reg_type=reg_type,
            prune_only=prune_only,
            fixed_8bit=fixed_8bit,
            fixed_4bit=fixed_48bit,
            fixed_full_precision=fixed_full_precision,
        )

        self.lora_B_quantizer = Quantizer(
            method,
            self.N,
            False,
            device=device,
            percentile=percentile,
            fix_range_on_first_forward=fix_range_on_first_forward,
            scale_domain=scale_domain,
            gating_method=gating_method,
            gamma_4_init=gamma_4_init,
            gamma_8_init=gamma_8_init,
            gamma_16_init=gamma_16_init,
            gamma_32_init=gamma_32_init,
            learned_scale=learned_scale,
            clip_input=clip_input,
            checkpointing=checkpointing,
            include_pruning=include_pruning,
            reg_type=reg_type,
            prune_only=prune_only,
            fixed_8bit=fixed_8bit,
            fixed_4bit=fixed_48bit,
            fixed_full_precision=fixed_full_precision,
        )
        self.lora_AB_act_quantizer = None
        self.lora_A_act_quantizer = None
        self.out_act_quantizer = None

        # self.lora_A_act_quantizer = Quantizer(
        #     method,
        #     self.n_bits_act,
        #     True,
        #     act_momentum,
        #     device=device,
        #     fix_range_on_first_forward=fix_range_on_first_forward,
        #     scale_domain=scale_domain,
        #     gating_method=gating_method,
        #     gamma_4_init=gamma_4_init,
        #     gamma_8_init=gamma_8_init,
        #     gamma_16_init=gamma_16_init,
        #     gamma_32_init=gamma_32_init,
        #     learned_scale=learned_scale,
        #     clip_input=clip_input,
        #     fixed_8bit=(fixed_8bit or fixed_48bit),
        #     act_quant=True,
        #     checkpointing=checkpointing,
        #     reg_type=reg_type,
        #     fixed_full_precision=fixed_full_precision,
        # )

        # self.lora_AB_act_quantizer = Quantizer(
        #     method,
        #     self.n_bits_act,
        #     True,
        #     act_momentum,
        #     device=device,
        #     fix_range_on_first_forward=fix_range_on_first_forward,
        #     scale_domain=scale_domain,
        #     gating_method=gating_method,
        #     gamma_4_init=gamma_4_init,
        #     gamma_8_init=gamma_8_init,
        #     gamma_16_init=gamma_16_init,
        #     gamma_32_init=gamma_32_init,
        #     learned_scale=learned_scale,
        #     clip_input=clip_input,
        #     fixed_8bit=(fixed_8bit or fixed_48bit),
        #     act_quant=True,
        #     checkpointing=checkpointing,
        #     reg_type=reg_type,
        #     fixed_full_precision=fixed_full_precision,
        # )

        # self.out_act_quantizer = Quantizer(
        #     method,
        #     self.n_bits_act,
        #     True,
        #     act_momentum,
        #     device=device,
        #     fix_range_on_first_forward=fix_range_on_first_forward,
        #     scale_domain=scale_domain,
        #     gating_method=gating_method,
        #     gamma_4_init=gamma_4_init,
        #     gamma_8_init=gamma_8_init,
        #     gamma_16_init=gamma_16_init,
        #     gamma_32_init=gamma_32_init,
        #     learned_scale=learned_scale,
        #     clip_input=clip_input,
        #     fixed_8bit=(fixed_8bit or fixed_48bit),
        #     act_quant=True,
        #     checkpointing=checkpointing,
        #     reg_type=reg_type,
        #     fixed_full_precision=fixed_full_precision,
        # )

    def quantize_activations(self, activations, quantizer):
        """Quantize a single activation tensor or all activations from a layer. I'm assuming that
        we should quantize all outputs for a layer with the same quantization scheme.
        """
        if self.activation_function is not None:
            activations = self.activation_function(activations)

        # if self.activation_save_target is not None:
        #     self.activation_save_target[self.activation_save_name] = (
        #         activations.data.cpu().numpy()
        #     )

        # if self._quant_a:
        #     activations = quantizer(activations)

        #     if self.activation_save_target is not None:
        #         self.activation_save_target[self.activation_save_name + "_Q"] = (
        #             activations.data.cpu().numpy()
        #         )
        return activations


class LoraQuantLinear(LoraQuantizationHijacker, lora_layers.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        r,
        reg_type="const",
        device="cuda",
        *args,
        activation=None,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r=r,
            reg_type=reg_type,
            activation=activation,
            merge_weights=False,
            device=device,
            *args,
            **kwargs,
        )

        # Simple training-time cache for quantized LoRA weights to avoid
        # re-quantizing on every forward when parameters are unchanged.
        self._wA_cached = None
        self._wB_cached = None
        self._w_cache_step = 0
        self._w_cache_steps = 4

    def get_lora_params(self):
        lora_A_weight, lora_B_weight = self.lora_A, self.lora_B

        if self._quant_w:
            # During training, reuse cached quantized weights for a few
            # steps to amortize quantization cost.
            if self.training:
                # Caching disabled to avoid double-backward error
                weight_A = self.lora_A_quantizer(lora_A_weight)
                weight_B = self.lora_B_quantizer(lora_B_weight)
            else:
                # In eval, quantize once and keep a persistent tensor cache
                if self.cached_params is not None:
                    return self.cached_params

                weight_A = self.lora_A_quantizer(lora_A_weight)
                weight_B = self.lora_B_quantizer(lora_B_weight)
                self.cached_params = (
                    torch.as_tensor(weight_A, device=weight_A.device),
                    torch.as_tensor(weight_B, device=weight_B.device),
                )
                return self.cached_params
        else:
            weight_A, weight_B = lora_A_weight, lora_B_weight

        return weight_A, weight_B

    def forward(self, x, offsets=None):
        weight, bias = self.get_weight_bias()
        weight_a, weight_b = self.get_lora_params()

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(weight), bias=bias)

            result = self.quantize_activations(result, self.activation_quantizer)

            lora_A_act = self.lora_dropout(x) @ weight_a.transpose(0, 1)
            lora_A_act = self.quantize_activations(
                lora_A_act, self.lora_A_act_quantizer
            )
            lora_act = lora_A_act @ weight_b.transpose(0, 1)
            lora_act = self.quantize_activations(lora_act, self.lora_AB_act_quantizer)
            result += lora_act * self.scaling

            result = self.quantize_activations(result, self.out_act_quantizer)
        else:
            result = F.linear(x, T(weight), bias=bias)
            result = self.quantize_activations(result, self.activation_quantizer)

        return result


class LoraSVDQuantLinear(LoraQuantizationHijacker, lora_layers.SVDLinear):
    def __init__(
        self,
        in_features,
        out_features,
        r,
        prune_rank=True,
        method="bayesian_bits",
        reg_type="const",
        act_momentum=0.1,
        *args,
        n_bits=16,
        percentile=None,
        fix_range_on_first_forward=False,
        scale_domain="linear",
        gating_method="l0",
        gamma_4_init=2.0,
        gamma_8_init=2.0,
        gamma_16_init=2.0,
        gamma_32_init=2.0,
        learned_scale=False,
        clip_input=False,
        checkpointing=False,
        include_pruning=False,
        prune_only=False,
        fixed_8bit=False,
        fixed_48bit=False,
        fixed_full_precision=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r=r,
            method=method,
            reg_type=reg_type,
            act_momentum=act_momentum,
            n_bits=n_bits,
            percentile=percentile,
            fix_range_on_first_forward=fix_range_on_first_forward,
            scale_domain=scale_domain,
            gating_method=gating_method,
            gamma_4_init=gamma_4_init,
            gamma_8_init=gamma_8_init,
            gamma_16_init=gamma_16_init,
            gamma_32_init=gamma_32_init,
            learned_scale=learned_scale,
            clip_input=clip_input,
            checkpointing=checkpointing,
            include_pruning=include_pruning,
            prune_only=prune_only,
            fixed_8bit=fixed_8bit,
            fixed_48bit=fixed_48bit,
            fixed_full_precision=fixed_full_precision,
            device=device,
            *args,
            **kwargs,
        )

        self.lora_E_quantizer = Quantizer(
            method,
            self.N,
            False,
            device=device,
            reg_type=reg_type,
            percentile=percentile,
            fix_range_on_first_forward=fix_range_on_first_forward,
            scale_domain=scale_domain,
            gating_method=gating_method,
            gamma_4_init=gamma_4_init,
            gamma_8_init=gamma_8_init,
            gamma_16_init=gamma_16_init,
            gamma_32_init=gamma_32_init,
            learned_scale=learned_scale,
            clip_input=clip_input,
            checkpointing=checkpointing,
            include_pruning=include_pruning,
            prune_only=prune_only,
            fixed_8bit=fixed_8bit,
            fixed_4bit=fixed_48bit,
            fixed_full_precision=fixed_full_precision,
            prune_rank=True,
        )
        self.lora_E_act_quantizer = None

        # self.lora_E_act_quantizer = Quantizer(
        #     method,
        #     self.n_bits_act,
        #     True,
        #     act_momentum,
        #     device=device,
        #     fix_range_on_first_forward=fix_range_on_first_forward,
        #     scale_domain=scale_domain,
        #     gating_method=gating_method,
        #     gamma_4_init=gamma_4_init,
        #     gamma_8_init=gamma_8_init,
        #     gamma_16_init=gamma_16_init,
        #     gamma_32_init=gamma_32_init,
        #     learned_scale=learned_scale,
        #     clip_input=clip_input,
        #     fixed_8bit=(fixed_8bit or fixed_48bit),
        #     act_quant=True,
        #     checkpointing=checkpointing,
        #     reg_type=reg_type,
        #     fixed_full_precision=fixed_full_precision,
        # )

        self.prune_rank = prune_rank

        # Simple cache for quantized LoRA weights to avoid excessive
        # recomputation; mirrors the behavior in LoraQuantLinear.
        # self._wA_cached = None
        # self._wB_cached = None
        # self._wE_cached = None
        # self._w_cache_step = 0
        # self._w_cache_steps = 4

    def get_lora_params(self):
        lora_A_weight, lora_B_weight, lora_E_weight = (
            self.lora_A,
            self.lora_B,
            self.lora_E,
        )
        if self._quant_w:
            if self.training:
                # Caching disabled to avoid double-backward error
                weight_A = self.lora_A_quantizer(lora_A_weight)
                weight_B = self.lora_B_quantizer(lora_B_weight)
                weight_E = self.lora_E_quantizer(lora_E_weight)
            else:
                if self.cached_params is not None:
                    wA, wB = self.cached_params
                    return (
                        wA,
                        wB,
                        (
                            self._wE_cached
                            if self._wE_cached is not None
                            else self.lora_E_quantizer(lora_E_weight)
                        ),
                    )

                weight_A = self.lora_A_quantizer(lora_A_weight)
                weight_B = self.lora_B_quantizer(lora_B_weight)
                weight_E = self.lora_E_quantizer(lora_E_weight)
                self.cached_params = (
                    torch.as_tensor(weight_A, device=weight_A.device),
                    torch.as_tensor(weight_B, device=weight_B.device),
                )
                self._wE_cached = torch.as_tensor(weight_E, device=weight_E.device)
                return (
                    self.cached_params[0],
                    self.cached_params[1],
                    self._wE_cached,
                )
        else:
            weight_A, weight_B, weight_E = (
                lora_A_weight,
                lora_B_weight,
                lora_E_weight,
            )

        return weight_A, weight_B, weight_E

    def forward(self, x, offsets=None):
        weight, bias = self.get_weight_bias()
        weight_a, weight_b, weight_e = self.get_lora_params()

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(weight), bias=bias)

            result = self.quantize_activations(result, self.activation_quantizer)

            lora_A_act = self.lora_dropout(x) @ weight_a.transpose(0, 1)
            lora_A_act = self.quantize_activations(
                lora_A_act, self.lora_A_act_quantizer
            ) * weight_e.transpose(0, 1)
            lora_A_act = self.quantize_activations(
                lora_A_act, self.lora_E_act_quantizer
            )
            lora_act = lora_A_act @ weight_b.transpose(0, 1)
            lora_act = self.quantize_activations(lora_act, self.lora_AB_act_quantizer)
            result += lora_act * self.scaling

            result = self.quantize_activations(result, self.out_act_quantizer)
        else:
            result = F.linear(x, T(weight), bias=bias)
            result = self.quantize_activations(result, self.activation_quantizer)

        return result


def quantize_apply_lora_model(
    model: nn.Module,
    model_name: str,
    lora_rank: int = 16,
    prune_rank: bool = False,
    lora_weights: tuple = (
        "query",
        "value",
        "key",
        "intermediate",
        "layer.output",
        "attention.output",
    ),
    quantize_lora_layers_only: bool = False,
    reg_type: str = "const",
    specials: dict | None = None,
    **quant_params,
):
    """
    Quantize model by substituting layers with their quantized versions.
        nn.Conv -> QuantConv
        nn.Linear -> QuantLinear
        nn.LoraLinear -> LoraQuantLinear

    :param model: model.
    :param model_name: name of the model.
    :param lora_rank: r parameter in LoRa decomposition.
    :param lora_weights: layers to apply LoRa to (any combination of keys, queries, values and attention output).
    :param quantize_lora_layers_only:  indicates if quantization is applied to LoRa matrices only.
    and context.
    :param specials: required for standard layers.
    :param quant_params: extra params.
    :return: quantized model.
    """
    specials = specials or dict()

    if isinstance(model, nn.Sequential):
        if not quantize_lora_layers_only:
            quant_model = quantize_sequential(model, specials, **quant_params)
        else:
            quant_model = model

    elif type(model) in specials:
        if not quantize_lora_layers_only:
            quant_model = specials[type(model)](model, **quant_params)
        else:
            quant_model = model

    elif type(model) in (nn.Conv2d, nn.Linear, lora_layers.Linear):
        modtype = None
        for lora_weight in lora_weights:
            if lora_weight in model_name:
                if prune_rank:
                    modtype = LoraSVDQuantLinear
                else:
                    modtype = LoraQuantLinear
                break
        if modtype is None and not quantize_lora_layers_only:
            modtype = non_bn_module_map[type(model)]

        if modtype is not None:
            kwargs = get_module_args(model, None)
            kwargs["include_pruning"] = False
            if modtype in (LoraQuantLinear, LoraSVDQuantLinear):
                kwargs["r"] = lora_rank
                kwargs["reg_type"] = reg_type
            quant_model = modtype(device=model.weight.device, **kwargs, **quant_params)

            quant_model.weight.data = model.weight.data
            if model.bias is not None:
                quant_model.bias.data = model.bias.data
            if modtype in (LoraQuantLinear, LoraSVDQuantLinear, QuantLinear):
                quant_model.quantized_weights()
            quant_model = quant_model.to(model.weight.device)
        else:
            quant_model = model

    else:
        quant_model = copy.copy(model)
        for name, module in quant_model._modules.items():
            new_model = quantize_apply_lora_model(
                module,
                model_name + "." + name,
                prune_rank=prune_rank,
                lora_rank=lora_rank,
                lora_weights=lora_weights,
                reg_type=reg_type,
                specials=specials,
                quantize_lora_layers_only=quantize_lora_layers_only,
                **quant_params,
            )
            if new_model is not None:
                setattr(quant_model, name, new_model)

    return quant_model
