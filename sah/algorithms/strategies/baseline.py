
from sah.algorithms.strategies.base_strategy import BaseStrategy


class BaselineStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.bits = 0

    def train_dataloader(self, pl_module):
        return None

    def configure_optimizers(self, pl_module):
        return None
