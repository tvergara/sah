from .base import BaseDatasetHandler, ProcessedTrainDataset


class OpenThoughtsHandler(BaseDatasetHandler):
    def __init__(self, tokenizer, dataset_name, block_size=1548, max_examples=None, val_size=12):
        super().__init__(tokenizer, dataset_name, block_size, max_examples)
        self.val_size = val_size

    def format_example(self, example):
        conversations = example['conversations']
        question = conversations[0]['value']
        answer = conversations[1]['value']
        return {"question": question, "answer": answer}

    def get_train_dataset(self):
        return ProcessedTrainDataset(
            self.tokenizer,
            self.dataset_name,
            self.format_example,
            self.block_size,
            max_examples=self.max_examples
        )

    def get_val_dataset(self):
        if self.max_examples is None:
            split_str = f"train[-{self.val_size}:]"
        else:
            split_str = f"train[{self.max_examples}:{self.max_examples + self.val_size}]"

        return ProcessedTrainDataset(
            self.tokenizer,
            self.dataset_name,
            self.format_example,
            self.block_size,
            split_str=split_str
        )

    def validate_batch(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss
        return {"performance": loss, "loss": loss}
