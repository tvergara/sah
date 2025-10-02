import torch
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.strategies.base_strategy import BaseStrategy


class IterativeStrategy(BaseStrategy):
    def __init__(self, lr, grads_in_memory):
        super().__init__()
        self.lr = lr
        self.grads_in_memory = grads_in_memory

    def setup(self, pl_module, stage):
        pass

    def compute_bits(self, pl_module):
        return 0

    def train_dataloader(self, pl_module):
        data_collator = DataCollatorForLanguageModeling(tokenizer=pl_module.tokenizer, mlm=False)
        sampler = SamplerWithSpecialFirstBatch(
            n=len(pl_module.dataset),
            first_batch_size=self.grads_in_memory,
            batch_size=pl_module.batch_size,
            shuffle=True
        )
        return torch.utils.data.DataLoader(
            pl_module.dataset,
            batch_sampler=sampler,
            num_workers=4,
            persistent_workers=True,
            collate_fn=data_collator,
        )



class SamplerWithSpecialFirstBatch:
    def __init__(self, n, first_batch_size, batch_size, shuffle=True):
        self.n = n
        self.first_batch_size = first_batch_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.randperm(self.n).tolist() if self.shuffle else list(range(self.n))
        yield indices[:self.first_batch_size]
        for i in range(self.first_batch_size, self.n, self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return 1 + (self.n - self.first_batch_size + self.batch_size - 1) // self.batch_size
