import glob
import math
import os
from dataclasses import dataclass

import hydra_zen
import pandas as pd
import torch
from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from sah.algorithms.networks.gaussian_transformer import GaussianTransformer


@hydra_zen.hydrated_dataclass(
    target=GaussianTransformer,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class TransformerConfig:
    embed_dim: int
    num_heads: int
    hidden_dim: int
    num_layers: int

@dataclass(frozen=True, unsafe_hash=True)
class GeneralConfig:
    result_file: str
    batch_size: int = 32
    lr: float = 1e-4
    test_size: float = 0.2

class ActivationDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        input_dir: str | None,
        output_dir: str,
        layers: list[int],
        mask_included: bool,
    ):
        super().__init__()
        self.layers = sorted(layers)
        inputs_per_layer = []
        outputs_per_layer = []
        masks_per_layer = []

        for L in self.layers:
            out_glob = os.path.join(base_path, output_dir, f"layer{L}_batch*.pt")
            out_paths = sorted(
                glob.glob(out_glob),
                key=lambda p: int(os.path.basename(p).split("_")[1].removeprefix("batch").removesuffix(".pt"))
            )[:100]
            if mask_included:
                out_tensors = []
                masks = []
                for path in out_paths:
                    values = torch.load(path)
                    activations = values['activations']
                    mask = values['mask']

                    if activations.shape[1] == 512:
                        out_tensors.append(activations)
                        masks.append(mask)
            else:
                out_tensors = [torch.load(p) for p in out_paths]

            if input_dir is None:
                inp_tensors = [torch.zeros_like(t) for t in out_tensors]
            else:
                inp_glob = os.path.join(base_path, input_dir, f"layer{L}_batch*.pt")
                inp_paths = sorted(
                    glob.glob(inp_glob),
                    key=lambda p: int(os.path.basename(p).split("_")[1].removeprefix("batch").removesuffix(".pt"))
                )
                inp_tensors = [torch.load(p) for p in inp_paths]

            inputs_per_layer.append(torch.cat(inp_tensors, dim=0))
            outputs_per_layer.append(torch.cat(out_tensors, dim=0))
            masks_per_layer.append(torch.cat(masks, dim=0))

        self.inputs  = torch.stack(inputs_per_layer,  dim=1)
        self.outputs = torch.stack(outputs_per_layer, dim=1)
        self.masks = torch.stack(masks_per_layer, dim=1)
        assert self.inputs.shape == self.outputs.shape

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.masks[idx], self.outputs[idx]

@hydra_zen.hydrated_dataclass(
    target=ActivationDataset,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class ActivationsConfig:
    base_path: str
    output_dir: str
    layers: list[int]
    input_dir: str | None = None
    mask_included: bool = False

class EntropyEstimator(LightningModule):
    def __init__(
        self,
        activations_config: ActivationsConfig,
        transformer_config: TransformerConfig,
        general_config: GeneralConfig,
    ) -> None:
        super().__init__()
        self.activations_config = activations_config
        self.transformer_config = transformer_config
        self.general_config = general_config
        self.network: GaussianTransformer | None = None
        self.loss_fn = nn.GaussianNLLLoss(full=True, eps=1e-5, reduction='none')

    def setup(self, stage: str):
        if stage == "fit":
            full_ds = hydra_zen.instantiate(self.activations_config)
            n = len(full_ds)
            test_n = int(self.general_config.test_size * n)
            train_n = n - test_n
            self.train_ds, self.test_ds = random_split(
                full_ds,
                [train_n, test_n],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.general_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.general_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def configure_model(self) -> None:
        if self.network is not None:
            return

        self.network = hydra_zen.instantiate(self.transformer_config)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int):
        input_acts, mask, output_acts = batch
        input_acts = input_acts.squeeze(1)
        output_acts = output_acts.squeeze(1)
        mask        = mask.squeeze(1).float()

        mu,  sigma  = self.network(input_acts)
        var = sigma.pow(2)

        per_elem_loss = self.loss_fn(mu, output_acts, var).mean(dim=-1)

        masked_loss = (per_elem_loss * mask).sum() / mask.sum()

        self.log("train_loss", masked_loss, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=mask.sum().item())

        return masked_loss

    def test_step(self, batch, batch_idx):
        input_acts, mask, output_acts = batch
        input_acts  = input_acts.squeeze(1)
        output_acts = output_acts.squeeze(1)
        mask        = mask.squeeze(1).bool()          # [B, L]

        mu, sigma = self.network(input_acts)
        var = sigma.square().clamp_min(1e-6)

        per_tok_nll = 0.5 * ( ((output_acts - mu).square() / var) +
                              torch.log(2 * math.pi * var) ).mean(dim=-1)

        avg_ent = per_tok_nll.masked_select(mask).mean()

        self.log("test/loss", avg_ent, on_epoch=True, prog_bar=True)
        return avg_ent

    def on_test_epoch_end(self):
        avg_ent = self.trainer.callback_metrics["test/loss"].item()
        self.save_entropy(avg_ent)

    def save_entropy(self, entropy):
        result_path = self.general_config.result_file
        input_dir = self.activations_config.input_dir
        output_dir = self.activations_config.output_dir
        row = {"input": input_dir, "output": output_dir, "entropy": entropy}

        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            if input_dir is None:
                mask = (df["input"].isna()) & (df["output"] == output_dir)
            else:
                mask = (df["input"] == input_dir) & (df["output"] == output_dir)
            if mask.any():
                df.loc[mask, "entropy"] = entropy
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row], columns=["input", "output", "entropy"])

        df.to_csv(result_path, index=False)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.network.parameters(),
            lr=self.general_config.lr,
        )
