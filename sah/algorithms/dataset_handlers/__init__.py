from .metamath import MetaMathHandler
from .openthoughts import OpenThoughtsHandler


def get_dataset_handler(dataset_name, tokenizer, block_size=1548, max_examples=None):
    if dataset_name == "open-thoughts/OpenThoughts3-1.2M":
        return OpenThoughtsHandler(tokenizer, dataset_name, block_size, max_examples)
    elif dataset_name == "meta-math/MetaMathQA":
        return MetaMathHandler(tokenizer, dataset_name, block_size, max_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
