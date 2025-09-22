import re
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import pandas as pd
import torch
from datasets import load_dataset
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig
from sah.algorithms.utils_file import load_weights_from_checkpoint


@dataclass(frozen=True, unsafe_hash=True)
class CheckpointConfig:
    path: str

class GSM8KDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.answers = []

        prompts = []
        for item in data:
            question = item['question']
            answer = item['answer']

            answer_match = re.search(r'#### ([\d,]+)', answer)
            numerical_answer = answer_match.group(1).replace(',', '') if answer_match else ""

            prompt = f"Question: {question}\nResponse:"
            prompts.append(prompt)
            self.answers.append(numerical_answer)

        # Tokenize all prompts at once
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
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

class EvaluateGSM8K(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        pretrained_config: NetworkConfig,
        checkpoint_config: CheckpointConfig,
        result_file: str,
        experiment_name: str,
        batch_size: int = 16,
        completion_length: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = hydra_zen.instantiate(tokenizer_config)
        self.model = hydra_zen.instantiate(pretrained_config, torch_dtype=torch.bfloat16)
        self.model.eval()
        load_weights_from_checkpoint(self.model, checkpoint_config.path, model_name='model')
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self.completion_length = completion_length
        self.result_file = Path(result_file)
        self.experiment_name = experiment_name


    def training_step(self, batch, batch_idx):
        pass

    def train_dataloader(self):
        pass

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.completion_length,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        correct_count = 0
        total_count = 0

        for i, gen_tokens in enumerate(generated):
            decoded = self.tokenizer.decode(gen_tokens[-self.completion_length:], skip_special_tokens=True)

            answer_patterns = [
                r'answer is: ([\d,]+)'
            ]

            extracted_answer = ""
            for pattern in answer_patterns:
                match = re.search(pattern, decoded, re.IGNORECASE)
                if match:
                    extracted_answer = match.group(1).replace(',', '')
                    break

            expected_answer = batch['expected_answer'][i]
            is_correct = extracted_answer == expected_answer

            if is_correct:
                correct_count += 1
            total_count += 1

        self.log("test/correct_count", float(correct_count), on_step=True, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("test/total_count", float(total_count), on_step=True, on_epoch=True, sync_dist=True, reduce_fx="sum")

        return {"correct_count": correct_count, "total_count": total_count}

    def test_dataloader(self):
        dataset = load_dataset("gsm8k", "main", split="test")

        gsm8k_dataset = GSM8KDataset(dataset, self.tokenizer)
        return DataLoader(gsm8k_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=3)

    def configure_optimizers(self):
        pass

    def on_test_epoch_end(self):
        correct_count = self.trainer.logged_metrics.get("test/correct_count_epoch", 0)
        total_count = self.trainer.logged_metrics.get("test/total_count_epoch", 1)
        correct_count = correct_count.item()
        total_count = total_count.item()

        accuracy = float(correct_count / total_count if total_count > 0 else 0.0)

        self.log("test/accuracy", accuracy, sync_dist=False)
        df = pd.read_csv(self.result_file)

        if "gsm8k" not in df.columns:
            df["gsm8k"] = None

        mask = df["experiment_name"] == self.experiment_name
        df.loc[mask, "gsm8k"] = accuracy
        df.to_csv(self.result_file, index=False)
