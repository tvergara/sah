from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


def listen_to_hidden_activations(
    model: nn.Module,
    activations_dict: dict[int, torch.Tensor],
    *,
    detach: bool = True,
    device: str | None = "cpu",
    selector=lambda m: isinstance(m, nn.TransformerEncoderLayer),
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    layer_counter = SimpleNamespace(idx=0)  # mutable int for closures

    for module in model.modules():
        if selector(module):
            idx = layer_counter.idx
            layer_counter.idx += 1

            def _hook(_, __, output, *, key=idx):
                if detach:
                    output = output.detach()
                if device is not None:
                    output = output.to(device)
                activations_dict[key] = output  # (B, L, d_model)

            h = module.register_forward_hook(_hook)
            handles.append(h)

    if not handles:
        raise RuntimeError(
            "No layers matched the selector — check your model or selector."
        )
    return handles
