import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForAnswerOnlyLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples]
        attention_mask = [torch.tensor(ex["attention_mask"]) for ex in examples]
        labels = [torch.tensor(ex["labels"]) for ex in examples]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100)
        }
