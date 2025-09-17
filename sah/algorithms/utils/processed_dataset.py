from datasets import load_dataset
from torch.utils.data import Dataset

from sah.algorithms.formatters import get_dataset_formatter


class ProcessedDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, block_size=512, max_examples=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.name = dataset_name

        print("Loading dataset...")
        raw_dataset = load_dataset(self.name, split="train")
        print(f"Dataset loaded: {len(raw_dataset)} examples")
        formatter = get_dataset_formatter(self.name)

        self.examples = []
        print("Processing examples...")
        total_to_process = min(len(raw_dataset), max_examples) if max_examples else len(raw_dataset)

        for i, example in enumerate(raw_dataset):
            if max_examples and i >= max_examples:
                break

            if i % 10000 == 0:
                print(f"Processed {i}/{total_to_process} examples")
            formatted = formatter(example)

            tokenized = tokenizer(
                formatted["text"],
                truncation=True,
                padding=False,
                max_length=block_size,
                return_tensors="pt"
            )

            if len(tokenized["input_ids"][0]) > 1:
                self.examples.append({
                    "input_ids": tokenized["input_ids"][0],
                    "attention_mask": tokenized["attention_mask"][0]
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
