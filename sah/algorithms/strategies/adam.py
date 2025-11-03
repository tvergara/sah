
import torch
import torch.distributed as dist

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.arithmetic_coding import compute_bits_from_logits


class AdamStrategy(BaseStrategy):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr
        self.bits = 0

    def configure_optimizers(self, pl_module):
        return torch.optim.Adam(pl_module.model.parameters(), lr=self.lr)

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss

        with torch.no_grad():
            total_bits = compute_bits_from_logits(
                outputs.logits,
                batch['input_ids'],
                batch.get('attention_mask')
            )

            if dist.is_initialized():
                total_bits_tensor = torch.tensor(total_bits, dtype=torch.float32, device=pl_module.device)
                dist.all_reduce(total_bits_tensor, op=dist.ReduceOp.SUM)
                total_bits = int(total_bits_tensor.item())

            self.bits += total_bits

        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
