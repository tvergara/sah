from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDatasetHandler:
    def __init__(self, tokenizer, dataset_name, block_size=1548, max_examples=None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.max_examples = max_examples
        self.validation_data = None

    def format_example(self, example):
        raise NotImplementedError

    def get_train_dataset(self):
        raise NotImplementedError

    def get_val_dataset(self):
        raise NotImplementedError

    def validate_batch(self, pl_module, batch, batch_idx):
        raise NotImplementedError

    def get_raw_val_data(self):
        raise NotImplementedError


class ProcessedTrainDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, format_fn, block_size=1548, max_examples=None, split_str=None):

        if split_str is None:
            split_str = f"train[:{max_examples}]" if max_examples else "train"

        raw_dataset = load_dataset(dataset_name, split=split_str)
        self.examples = []

        for example in tqdm(raw_dataset):
            formatted = format_fn(example)
            question_text = f"Question: {formatted['question']}\nResponse:"
            answer_text = " " + formatted['answer']

            question_ids = tokenizer.encode(question_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

            if tokenizer.bos_token_id is not None:
                full_ids = [tokenizer.bos_token_id] + question_ids + answer_ids
                question_length_offset = 1
            else:
                full_ids = question_ids + answer_ids
                question_length_offset = 0

            if len(full_ids) > block_size:
                full_ids = full_ids[:block_size]

            question_length = min(question_length_offset + len(question_ids), len(full_ids))

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


class GenerationValDataset(Dataset):
    def __init__(self, input_ids, attention_masks, answers):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.answers = answers

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'expected_answer': self.answers[idx]
        }
