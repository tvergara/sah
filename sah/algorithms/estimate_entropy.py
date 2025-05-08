import glob
import os
from dataclasses import dataclass

import hydra_zen
import torch
from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.networks.gaussian_transformer import GaussianTransformer


@dataclass(frozen=True, unsafe_hash=True)
class ActivationsConfig:
    activations_path: str
    input_activations: str
    output_activations: str
    layers: list[int]
    batch_size: int = 32
    lr: float = 1e-4

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

class ActivationDataset(Dataset):
    def __init__(self, base_path: str, input_dir: str, output_dir: str, layers: list[int]):
        super().__init__()
        self.layers = sorted(layers)
        inputs_per_layer = []
        outputs_per_layer = []

        for L in self.layers:
            inp_glob = os.path.join(base_path, input_dir,  f"layer{L}_batch*.pt")
            out_glob = os.path.join(base_path, output_dir, f"layer{L}_batch*.pt")

            def sorted_by_batch(pattern):
                files = glob.glob(pattern)

                def batch_idx(path):
                    name = os.path.basename(path)
                    return int(name.split("_")[1].removeprefix("batch").removesuffix(".pt"))
                return sorted(files, key=batch_idx)

            inp_paths = sorted_by_batch(inp_glob)
            out_paths = sorted_by_batch(out_glob)

            inp_tensors = [torch.load(p) for p in inp_paths]
            out_tensors = [torch.load(p) for p in out_paths]

            inputs_per_layer.append(torch.cat(inp_tensors, dim=0))
            outputs_per_layer.append(torch.cat(out_tensors, dim=0))

        self.inputs  = torch.stack(inputs_per_layer,  dim=1)
        self.outputs = torch.stack(outputs_per_layer, dim=1)

        assert self.inputs.shape == self.outputs.shape

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class EntropyEstimator(LightningModule):
    def __init__(
        self,
        activations_config: ActivationsConfig,
        transformer_config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.activations_config = activations_config
        self.transformer_config = transformer_config
        self.network: GaussianTransformer | None = None
        self.loss_fn = nn.GaussianNLLLoss(full=True, eps=1e-6)

    def train_dataloader(self):
        ds = ActivationDataset(
            base_path=self.activations_config.activations_path,
            input_dir=self.activations_config.input_activations,
            output_dir=self.activations_config.output_activations,
            layers=self.activations_config.layers,
        )
        return DataLoader(
            ds,
            batch_size=self.activations_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def configure_model(self) -> None:
        if self.network is not None:
            return

        self.network = hydra_zen.instantiate(self.transformer_config)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int):
        input_acts, output_acts = batch
        input_acts = input_acts.squeeze(1)
        output_acts = output_acts.squeeze(1)
        mu, sigma = self.network(input_acts)
        var = sigma ** 2
        loss = self.loss_fn(mu, output_acts, var)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.network.parameters(),
            lr=self.activations_config.lr,
        )
