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

            question_text = f"Question: {formatted['question']}\nResponse:"
            answer_text = " " + formatted['answer']

            question_ids = tokenizer.encode(question_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

            full_ids = [tokenizer.bos_token_id] + question_ids + answer_ids
            if len(full_ids) > block_size:
                full_ids = full_ids[:block_size]

            question_length = 1 + len(question_ids)

            if len(full_ids) > question_length + 1:
                labels = full_ids.copy()
                labels[:question_length] = [-100] * question_length

                self.examples.append({
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
