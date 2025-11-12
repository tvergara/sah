import sacrebleu
import torch
from datasets import load_dataset

from .base import BaseDatasetHandler, GenerationValDataset, ProcessedTrainDataset


class FLORESHandler(BaseDatasetHandler):
    def __init__(
        self,
        tokenizer,
        dataset_name="allenai/nllb",
        block_size=1548,
        max_examples=None,
        train_config="eng_Latn-fra_Latn",
        eval_dataset="facebook/flores",
        eval_config="eng_Latn-fra_Latn",
        eval_split="devtest"
    ):
        super().__init__(tokenizer, dataset_name, block_size, max_examples)
        self.train_config = train_config
        self.eval_dataset = eval_dataset
        self.eval_config = eval_config
        self.eval_split = eval_split

    def format_example(self, example):
        if 'translation' in example:
            translation = example['translation']
            english = translation['eng_Latn']
            french = translation['fra_Latn']
        else:
            english = example['sentence_eng_Latn']
            french = example['sentence_fra_Latn']

        return {
            "question": f"Translate to French: {english}",
            "answer": french
        }

    def get_train_dataset(self):
        return ProcessedTrainDataset(
            self.tokenizer,
            self.dataset_name,
            self.format_example,
            self.block_size,
            max_examples=self.max_examples,
            split_str="train",
            config_name=self.train_config,
            streaming=True
        )

    def get_val_dataset(self):
        raw_dataset = load_dataset(self.eval_dataset, self.eval_config, split=self.eval_split, trust_remote_code=True)
        prompts = []
        answers = []

        for item in raw_dataset:
            formatted = self.format_example(item)
            prompt = f"{formatted['question']}\nTranslation:"
            prompts.append(prompt)
            answers.append(formatted['answer'])

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.block_size,
            padding_side='left',
        )

        return GenerationValDataset(
            tokenized['input_ids'],
            tokenized['attention_mask'],
            answers
        )

    def get_raw_val_data(self):
        if self.validation_data is None:
            raw_dataset = load_dataset(self.eval_dataset, self.eval_config, split=self.eval_split, trust_remote_code=True)
            prompts = []
            answers = []

            for item in raw_dataset:
                formatted = self.format_example(item)
                prompt = f"{formatted['question']}\nTranslation:"
                prompts.append(prompt)
                answers.append(formatted['answer'])

            self.validation_data = [
                {"question": q, "expected_answer": a}
                for q, a in zip(prompts, answers)
            ]

        return self.validation_data

    def validate_batch(self, pl_module, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            generated = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=pl_module.max_length if hasattr(pl_module, 'max_length') else 128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoding_starts = batch['input_ids'].shape[1]
        hypotheses = []
        references = []

        for i, gen_tokens in enumerate(generated):
            decoded = self.tokenizer.decode(
                gen_tokens[decoding_starts:],
                skip_special_tokens=True
            ).strip()

            if '\n' in decoded:
                decoded = decoded.split('\n')[0].strip()

            hypotheses.append(decoded)
            references.append(batch['expected_answer'][i])

        bleu = sacrebleu.corpus_bleu(hypotheses, [references])

        return {
            "performance": float(bleu.score),
            "bleu_score": float(bleu.score),
            "total_count": float(len(hypotheses))
        }
