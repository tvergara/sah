from datasets import load_dataset
from torch.utils.data import Dataset


class BaseDatasetHandler:
    def __init__(
        self,
        tokenizer,
        dataset_name,
        block_size=1548,
        max_examples=None,
        generations_dir=None,
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.max_examples = max_examples
        self.generations_dir = generations_dir
        self.validation_data = None
        self.generations = []

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
    def __init__(
        self,
        tokenizer,
        dataset_name,
        format_fn,
        block_size=1548,
        max_examples=None,
        split_str=None,
        config_name=None,
        streaming=False,
    ):

        if split_str is None:
            split_str = f"train[:{max_examples}]" if max_examples else "train"

        if config_name is not None:
            raw_dataset = load_dataset(
                dataset_name,
                config_name,
                split=split_str,
                streaming=streaming,
                trust_remote_code=True,
            )
        else:
            raw_dataset = load_dataset(
                dataset_name,
                split=split_str,
                streaming=streaming,
                trust_remote_code=True,
            )
        assert not isinstance(raw_dataset, dict)

        if streaming and max_examples:
            raw_dataset = raw_dataset.take(max_examples)

        process_fn = create_process_fn(tokenizer, format_fn, block_size)

        map_kwargs = {
            "remove_columns": raw_dataset.column_names,
            "batched": True,
        }
        if not streaming:
            map_kwargs["num_proc"] = 8

        self.dataset = raw_dataset.map(process_fn, **map_kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_process_fn(tokenizer, format_fn, block_size):
    def process_fn(examples):
        processed = {"input_ids": [], "attention_mask": [], "labels": []}
        keys = examples.keys()

        for example in [dict(zip(keys, values)) for values in zip(*examples.values())]:
            formatted = format_fn(example)
            question_text = formatted["question"]
            answer_text = formatted["answer"]

            question_ids = tokenizer.encode(question_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

            if tokenizer.bos_token_id is not None:
                full_ids = [tokenizer.bos_token_id] + question_ids + answer_ids
                question_length = 1 + len(question_ids)
            else:
                full_ids = question_ids + answer_ids
                question_length = len(question_ids)

            if len(full_ids) > block_size:
                full_ids = full_ids[:block_size]

            if len(full_ids) > question_length:
                labels = full_ids.copy()
                labels[:question_length] = [-100] * question_length

                processed["input_ids"].append(full_ids)
                processed["attention_mask"].append([1] * len(full_ids))
                processed["labels"].append(labels)
        return processed

    return process_fn


class GenerationValDataset(Dataset):
    def __init__(self, input_ids, attention_masks, answers):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.answers = answers

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "expected_answer": self.answers[idx],
        }
