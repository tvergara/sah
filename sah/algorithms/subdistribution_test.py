

import pickle
from dataclasses import dataclass

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader

from sah.algorithms.networks.transformer import TransformerConfig
from sah.algorithms.utils import (
    GrammarConfig,
    TokenizerConfig,
    collate,
    load_weights_from_checkpoint,
)


@dataclass(frozen=True, unsafe_hash=True)
class GeneralConfig:
    distribution_model: str
    subdistribution_model: str

class SubdistributionTest(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        transformer_config: TransformerConfig,
        general_config: GeneralConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grammar_config = grammar_config
        with open(tokenizer_config.out_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.test_dataset = hydra_zen.instantiate(
            self.grammar_config,
            mode='test',
            tokenizer=self.tokenizer,
            size=None
        )
        self.pad  = self.test_dataset.pad_id
        self.general_config = general_config

        self.distribution_model = hydra_zen.instantiate(
            transformer_config,
            vocab_size = self.tokenizer.vocab_size,
        )
        load_weights_from_checkpoint(
            self.distribution_model,
            general_config.distribution_model,
            model_name='transformer'
        )

        self.subdistribution_model = hydra_zen.instantiate(
            transformer_config,
            vocab_size = self.tokenizer.vocab_size,
        )
        load_weights_from_checkpoint(
            self.subdistribution_model,
            general_config.subdistribution_model,
            model_name='transformer'
        )

    def training_step(self, batch, batch_idx):
        pass
    def train_dataloader(self):
        pass
    def configure_optimizers(self):
        pass

    def test_step(self, batch, batch_idx):
        x, _ = batch
        inputs  = x[:, :-1]

        distribution_logits = self.distribution_model(inputs)
        subdistribution_logits = self.subdistribution_model(inputs)

        cross_entropy = self.cross_entropy(distribution_logits, subdistribution_logits)
        entropy = self.cross_entropy(distribution_logits, distribution_logits)

        self.log("distance", cross_entropy, prog_bar=True, sync_dist=True)
        self.log("entropy diff", entropy, prog_bar=True, sync_dist=True)

    def cross_entropy(self, distribution_logits, subdistribution_logits):
        log_probs_distribution = F.log_softmax(distribution_logits, dim=-1)
        probs_subdistribution = F.softmax(subdistribution_logits, dim=-1)

        cross_entropy = -(probs_subdistribution * log_probs_distribution).sum(dim=-1)

        mean_cross_entropy = cross_entropy.mean()
        return mean_cross_entropy

    def jensen_shannon_distance(self, logits_p, logits_q):
        probs_p = F.softmax(logits_p, dim=-1)  # P
        probs_q = F.softmax(logits_q, dim=-1)  # Q

        m = 0.5 * (probs_p + probs_q)

        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        log_m = torch.log(m + 1e-8)  # add epsilon for numerical stability

        kl_pm = (probs_p * (log_p - log_m)).sum(dim=-1)  # KL(P || M)
        kl_qm = (probs_q * (log_q - log_m)).sum(dim=-1)  # KL(Q || M)

        js_divergence = 0.5 * (kl_pm + kl_qm)  # shape: (B,)
        js_divergence = F.relu(js_divergence)
        js_distance = torch.sqrt(js_divergence + 1e-8)   # small epsilon to avoid sqrt(0)

        return js_distance.mean()

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, collate_fn=collate)
