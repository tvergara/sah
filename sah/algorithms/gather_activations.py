"""Script to gather activations from multiple model checkpoints.

This script is similar to llm_finetuning.py but focuses on running inference and saving activations
rather than training. It processes a dataset through multiple model checkpoints and saves the
residual stream activations for analysis.
"""

import dataclasses
import hashlib
import itertools
import os
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Concatenate, ParamSpec

import datasets
import datasets.distributed
import hydra_zen
import numpy as np
import torch
import torch.distributed
from datasets import Dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerBase

from sah.algorithms.formatters import get_dataset_formatter
from sah.utils.env_vars import SCRATCH, SLURM_TMPDIR

logger = getLogger(__name__)

def num_cpus_per_task() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()

@hydra_zen.hydrated_dataclass(
    target=AutoModelForCausalLM.from_pretrained,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class NetworkConfig:
    """Configuration options related to the choice of network."""
    pretrained_model_name_or_path: str
    revision: str
    trust_remote_code: bool = False
    torch_dtype: torch.dtype | None = None

@hydra_zen.hydrated_dataclass(
    target=AutoTokenizer.from_pretrained,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class TokenizerConfig:
    """Configuration options for the tokenizer."""
    pretrained_model_name_or_path: str
    cache_dir: Path | None = None
    force_download: bool = False
    local_files_only: bool = False
    token: str | bool | None = None
    revision: str = "main"
    use_fast: bool = True
    config: PretrainedConfig | None = None
    subfolder: str = ""
    tokenizer_type: str | None = None
    trust_remote_code: bool = False

@dataclass(frozen=True, unsafe_hash=True)
class DatasetConfig:
    """Configuration options related to the dataset preparation."""
    dataset_path: str
    dataset_name: str | None = None
    per_device_eval_batch_size: int = dataclasses.field(
        default=8, metadata={"include_in_id": False}
    )
    block_size: int = 1024
    preprocessing_num_workers: int = num_cpus_per_task()
    validation_split_percentage: int = 10
    overwrite_cache: bool = False

@dataclass(frozen=True, unsafe_hash=True)
class SavingConfig:
    output_dir: str

def load_raw_datasets(config: DatasetConfig):
    raw_datasets = datasets.load_dataset(config.dataset_path, config.dataset_name)
    assert isinstance(raw_datasets, DatasetDict)
    if "validation" not in raw_datasets.keys() and config.validation_split_percentage > 0:
        raw_datasets["validation"] = datasets.load_dataset(
            config.dataset_path,
            config.dataset_name,
            split=f"train[:{config.validation_split_percentage}%]",
        )
        raw_datasets["train"] = datasets.load_dataset(
            config.dataset_path,
            config.dataset_name,
            split=f"train[{config.validation_split_percentage}%:]",
        )
    return raw_datasets

def prepare_datasets(
    dataset_config: DatasetConfig, tokenizer_config: TokenizerConfig
) -> DatasetDict:
    raw_datasets = load_raw_datasets(dataset_config)
    tokenizer = load_tokenizer(tokenizer_config)
    formatter = get_dataset_formatter(dataset_config.dataset_path)
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, dataset_config, formatter)
    lm_datasets = group_text_into_blocks(tokenized_datasets, tokenizer, dataset_config)
    return lm_datasets

def load_tokenizer(config: TokenizerConfig) -> PreTrainedTokenizerBase:
    return hydra_zen.instantiate(config)

def tokenize_datasets(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: DatasetConfig,
    formatter: Callable,
) -> DatasetDict:
    formatted_dataset = raw_datasets.map(formatter)
    return formatted_dataset.map(
        lambda b: tokenizer(b["text"]),
        batched=True,
        remove_columns=formatted_dataset["train"].column_names,
        load_from_cache_file=not config.overwrite_cache,
        desc="Tokenizing the dataset",
    )

def group_text_into_blocks(
    tokenized_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: DatasetConfig,
) -> DatasetDict:
    block_size = config.block_size
    if block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
        block_size = tokenizer.model_max_length

    return tokenized_datasets.map(
        group_texts,
        fn_kwargs={"block_size": block_size},
        batched=True,
        load_from_cache_file=True,
        num_proc=config.preprocessing_num_workers,
        desc=f"Grouping tokens into chunks of size {block_size}",
    )

def group_texts(examples: dict, block_size: int):
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

class ActivationGatherer(LightningModule):
    """Module for gathering activations from multiple model checkpoints."""

    def __init__(
        self,
        network_config: NetworkConfig,
        tokenizer_config: TokenizerConfig,
        dataset_config: DatasetConfig,
        saving_config: SavingConfig,
        seed: int,
    ):
        super().__init__()

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.network_config = network_config
        self.tokenizer_config = tokenizer_config
        self.dataset_config = dataset_config
        self.saving_config = saving_config

        self.save_hyperparameters(
            dict(
                network_config=dataclasses.asdict(network_config),
                tokenizer_config=dataclasses.asdict(tokenizer_config),
                dataset_config=dataclasses.asdict(dataset_config),
            )
        )

        self.prepare_data_per_node = True
        self.data_configs_id = (
            f"{get_hash_of(self.dataset_config)[:8]}_{get_hash_of(self.tokenizer_config)[:8]}"
        )
        logger.info(f"Unique id for our dataset / tokenizer configs: {self.data_configs_id}")

        self.scratch_prepared_dataset_dir: Path | None = None
        if SCRATCH is not None:
            self.scratch_prepared_dataset_dir = (
                SCRATCH / "data" / "prepared_dataset" / self.data_configs_id
            )
            self.scratch_prepared_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        fast_data_dir = (SLURM_TMPDIR or Path.cwd()) / "data" / "prepared_dataset"
        self.fast_prepared_dataset_dir = fast_data_dir / self.data_configs_id
        self.fast_prepared_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.valid_dataset: Dataset | None = None
        self.network: AutoModelForCausalLM | None = None

    def prepare_data(self):
        if _try_to_load_prepared_dataset_from(self.fast_prepared_dataset_dir):
            logger.info(
                f"Dataset is already prepared on this node at {self.fast_prepared_dataset_dir}"
            )
            return
        logger.debug("Dataset hasn't been prepared on this node yet.")

        if not self.scratch_prepared_dataset_dir:
            assert self.trainer.num_nodes == 1
            logger.info(f"Preparing dataset at {self.fast_prepared_dataset_dir}.")
            lm_datasets = prepare_datasets(self.dataset_config, self.tokenizer_config)
            lm_datasets.save_to_disk(self.fast_prepared_dataset_dir)
            return

        if _try_to_load_prepared_dataset_from(self.scratch_prepared_dataset_dir):
            logger.info(
                f"Dataset is already prepared on the shared filesystem at "
                f"{self.scratch_prepared_dataset_dir}"
            )
            copy_dataset_files(self.scratch_prepared_dataset_dir, self.fast_prepared_dataset_dir)
            return

        logger.debug("Dataset has not yet been prepared with this config yet.")

        if self.trainer.num_nodes == 1:
            logger.debug("Single-node training. Preparing the dataset.")
            lm_datasets = prepare_datasets(self.dataset_config, self.tokenizer_config)
            lm_datasets.save_to_disk(self.fast_prepared_dataset_dir)
            logger.info(f"Saved processed dataset to {self.fast_prepared_dataset_dir}")
            copy_dataset_files(self.fast_prepared_dataset_dir, self.scratch_prepared_dataset_dir)
            return

        _barrier_name = "prepare_data"
        if self.global_rank == 0:
            logger.info(
                f"Task {self.global_rank}: Preparing the dataset in $SLURM_TMPDIR and copying it to $SCRATCH."
            )
            lm_datasets = prepare_datasets(self.dataset_config, self.tokenizer_config)
            lm_datasets.save_to_disk(self.fast_prepared_dataset_dir)
            logger.info(f"Saved processed dataset to {self.fast_prepared_dataset_dir}")
            copy_dataset_files(self.fast_prepared_dataset_dir, self.scratch_prepared_dataset_dir)
            logger.info(f"Task {self.global_rank}: Done preparing the dataset.")
            self.trainer.strategy.barrier(_barrier_name)
        else:
            logger.info(
                f"Task {self.global_rank}: Waiting for the first task on the first node to finish preparing the dataset."
            )
            self.trainer.strategy.barrier(_barrier_name)
            assert self.scratch_prepared_dataset_dir.exists()
            logger.info(
                f"Copying the dataset files prepared by the first node at {self.scratch_prepared_dataset_dir}"
            )
            copy_dataset_files(self.scratch_prepared_dataset_dir, self.fast_prepared_dataset_dir)

        logger.info(f"Done preparing the datasets at {self.fast_prepared_dataset_dir}.")

    def setup(self, stage: str):
        self.tokenizer = load_tokenizer(self.tokenizer_config)
        lm_datasets = datasets.load_from_disk(self.fast_prepared_dataset_dir)
        logger.info(f"Loading processed dataset from {self.fast_prepared_dataset_dir}")
        assert isinstance(lm_datasets, DatasetDict)
        self.valid_dataset = lm_datasets["validation"]

    def configure_model(self) -> None:
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization
        # Initialize the weights on the GPU if we have one, so we don't
        # request lots of RAM just to load up the model weights and then not use it.
        if self.network is not None:
            return
        logger.info(f"Rank {self.local_rank}: {self.device=}")
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            self.network = hydra_zen.instantiate(self.network_config)
            self.network.eval()

    def train_dataloader(self):
        assert self.valid_dataset is not None
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            collate_fn=default_data_collator,
            num_workers=self.dataset_config.preprocessing_num_workers,
            batch_size=self.dataset_config.per_device_eval_batch_size,
            worker_init_fn=lambda worker_id: torch.manual_seed(self.seed + worker_id),
            generator=g,
            pin_memory=True,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        with torch.no_grad():
            outputs: CausalLMOutput = self.network(
                **batch,
                output_hidden_states=True
            )

        for layer_idx, h in enumerate(outputs.hidden_states):
            tensor = h.cpu()
            torch.save(
                tensor,
                self.saving_config.output_dir + f"/layer{layer_idx}_batch{batch_idx}.pt"
            )
            del tensor

    def configure_optimizers(self):
        pass

def copy_dataset_files(src: Path, dest: Path):
    logger.info(f"Copying dataset from {src} --> {dest}")
    shutil.copytree(src, dest)

P = ParamSpec("P")

def _try_to_load_prepared_dataset_from(
    dataset_path: Path,
    _load_from_disk_fn: Callable[Concatenate[Path, P], Dataset | DatasetDict] = load_from_disk,
    *_load_from_disk_args: P.args,
    **_load_from_disk_kwargs: P.kwargs,
) -> DatasetDict | None:
    try:
        datasets = _load_from_disk_fn(
            dataset_path, *_load_from_disk_args, **_load_from_disk_kwargs
        )
    except FileNotFoundError as exc:
        logger.debug(f"Unable to load the prepared dataset from {dataset_path}: {exc}")
        return None
    else:
        logger.debug(f"Dataset is already prepared at {dataset_path}")
        assert isinstance(datasets, DatasetDict)
        return datasets

def get_hash_of(obj: object) -> str:
    """Get a hash of an object's state."""
    return hashlib.md5(str(obj).encode()).hexdigest()
