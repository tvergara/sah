from .flores import FLORESHandler
from .lima import LimaHandler
from .metamath import MetaMathHandler
from .mmlu import MMLUHandler
from .openthoughts import OpenThoughtsHandler
from .piqa import PiQAHandler


def get_dataset_handler(dataset_name, tokenizer, block_size=1548, max_examples=None, generations_dir=None):
    if dataset_name == "open-thoughts/OpenThoughts3-1.2M":
        return OpenThoughtsHandler(tokenizer, dataset_name, block_size, max_examples)
    elif dataset_name == "meta-math/MetaMathQA":
        return MetaMathHandler(tokenizer, dataset_name, block_size, max_examples)
    elif dataset_name == "cais/mmlu" or dataset_name.startswith("mmlu"):
        return MMLUHandler(tokenizer, dataset_name, block_size, max_examples)
    elif dataset_name == "allenai/nllb" or dataset_name.startswith("flores"):
        return FLORESHandler(tokenizer, dataset_name, block_size, max_examples)
    elif dataset_name == "ybisk/piqa" or dataset_name.startswith("piqa"):
        return PiQAHandler(tokenizer, dataset_name, block_size, max_examples)
    elif dataset_name == "GAIR/lima" or dataset_name.startswith("lima"):
        return LimaHandler(tokenizer, dataset_name, block_size, max_examples, generations_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
