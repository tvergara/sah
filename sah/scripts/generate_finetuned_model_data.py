from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import hydra_zen
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig
from sah.algorithms.utils import load_weights_from_checkpoint

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCfg:
    tokenizer_config: TokenizerConfig = field(default_factory=lambda: TokenizerConfig(
        pretrained_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        use_fast=True,
        trust_remote_code=True,
    ))
    finetuned_config: NetworkConfig = field(default_factory=lambda: NetworkConfig(
        pretrained_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    ))
    checkpoint_config: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(
        path="/network/scratch/b/brownet/hydra-runs/finetune-on-lima/checkpoints/finetuned",
    ))


@dataclass
class DatasetCfg:
    n_batches: int = 1000
    batch_size: int = 32
    seq_length: int = 100
    save_dir: str = "finetuned-model-data"
    seed: int = 42


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)


@dataclass(frozen=True, unsafe_hash=True)
class CheckpointConfig:
    path: str

# ──────────────────────────────────────────────────────────────────────────────
# Data Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_sequences_batch(
    model,
    tokenizer,
    batch_size: int,
    seq_length: int,
) -> list[list[int]]:
    """Generate a batch of sequences from the finetuned model."""
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    # Initialize batch with random starting tokens
    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
    input_ids[:, 0] = torch.randint(1, vocab_size - 1, (batch_size,), device=device)

    # Generate sequences token by token
    with torch.no_grad():
        for i in range(1, seq_length):
            # Get logits for all sequences at position i-1
            outputs = model(input_ids[:, :i])
            logits = outputs.logits[:, -1]  # [batch_size, vocab_size]

            # Sample next tokens
            dist = torch.distributions.Categorical(logits=logits)
            next_tokens = dist.sample()  # [batch_size]
            input_ids[:, i] = next_tokens

    # Convert to list of lists of integers
    return input_ids.cpu().tolist()


def create_dataset(cfg: Config) -> None:
    """Generate synthetic data from finetuned model and save in batches."""
    print("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = hydra_zen.instantiate(cfg.model.tokenizer_config)
    tokenizer = hydra_zen.instantiate(tokenizer)

    # Load finetuned model
    model = hydra_zen.instantiate(cfg.model.finetuned_config, torch_dtype=torch.bfloat16)
    model = hydra_zen.instantiate(model, device_map='auto')
    load_weights_from_checkpoint(model, cfg.model.checkpoint_config.path, model_name='model')
    model.eval()
    if device == "cpu":
        model = model.to(device)

    # Set random seed
    random.seed(cfg.dataset.seed)
    torch.manual_seed(cfg.dataset.seed)

    # Create output directory
    out_dir = Path(cfg.dataset.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    (out_dir / "hyperparams.yaml").write_text(OmegaConf.to_yaml(cfg))

    print(f"Generating {cfg.dataset.n_batches} batches of {cfg.dataset.batch_size} sequences each...")
    print(f"Sequence length: {cfg.dataset.seq_length}")
    print(f"Total sequences: {cfg.dataset.n_batches * cfg.dataset.batch_size}")

    # Generate and save batches
    for batch_idx in tqdm(range(cfg.dataset.n_batches), desc="Generating batches"):
        # Generate batch of sequences
        sequences = generate_sequences_batch(
            model=model,
            tokenizer=tokenizer,
            batch_size=cfg.dataset.batch_size,
            seq_length=cfg.dataset.seq_length,
        )

        # Save batch immediately to avoid memory issues
        batch_file = out_dir / f"batch_{batch_idx+1:04d}.txt"
        with batch_file.open("w") as f:
            for sequence in sequences:
                # Save as space-separated token IDs
                f.write(" ".join(map(str, sequence)) + "\n")

    print(
        f"\n✨ Saved dataset to {out_dir.resolve()}\n"
        f"   ├─ {cfg.dataset.n_batches} batch files (batch_0001.txt - batch_{cfg.dataset.n_batches:04d}.txt)\n"
        f"   ├─ {cfg.dataset.n_batches * cfg.dataset.batch_size} total sequences\n"
        f"   └─ hyperparams.yaml (configuration snapshot)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs/scripts", config_name="generate_finetuned_model_data")
def main(cfg: Config):  # type: ignore[arg-type]
    print("Hydra configuration (active):\n" + OmegaConf.to_yaml(cfg))
    create_dataset(cfg)


if __name__ == "__main__":
    main()
