import glob
import os
from dataclasses import dataclass

import torch
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True, unsafe_hash=True)
class ActivationsConfig:
    activations_path: str
    input_activations: str
    output_activations: str
    layers: list[int]
    batch_size: int = 32


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
    def __init__(self, activations_config: ActivationsConfig) -> None:
        super().__init__()
        self.activations_config = activations_config

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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        pass

    def configure_optimizers(self):
        pass
