
import re

from datasets import load_dataset
from torch.utils.data import Dataset


class ProcessedValidationDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, block_size=1548, max_examples=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.name = dataset_name
        self.answers = []

        raw_dataset = load_dataset(*self.name, split="test")
        prompts = []
        for item in raw_dataset:
            question = item['question']
            answer = item['answer']

            answer_match = re.search(r'#### ([\d,]+)', answer)
            numerical_answer = answer_match.group(1).replace(',', '') if answer_match else ""

            prompt = f"Question: {question}\nResponse:"
            prompts.append(prompt)
            self.answers.append(numerical_answer)

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=block_size,
            padding_side='left',
        )

        self.input_ids = tokenized['input_ids']
        self.attention_masks = tokenized['attention_mask']

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'expected_answer': self.answers[idx]
        }
