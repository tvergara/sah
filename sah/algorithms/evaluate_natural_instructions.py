import re
from dataclasses import dataclass

import hydra_zen
import requests
import torch
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig

from .utils import load_weights_from_checkpoint


@dataclass(frozen=True, unsafe_hash=True)
class CheckpointConfig:
    path: str

class NaturalInstructionsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        definition = data['Definition'][0]
        self.prompts = []
        self.outputs = []

        for instance in data['Instances']:
            entry = instance['input']
            output = instance['output'][0]

            prompt = f"{definition} You are allowed to think, and then you will answer by saying 'the answer is (1 or 0)'. The date you will use {entry}, is it a valid date? Please reason and then answer\n\nAssistant: First,"

            self.prompts.append(prompt)
            self.outputs.append(output)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        output = self.outputs[idx]

        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'prompt': prompt,
            'expected_output': output
        }

class EvaluateNaturalInstructions(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        pretrained_config: NetworkConfig,
        checkpoint_config: CheckpointConfig,
        batch_size: int = 16,
        completion_length: int = 512
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = hydra_zen.instantiate(tokenizer_config)
        self.model = hydra_zen.instantiate(pretrained_config, torch_dtype=torch.bfloat16)
        load_weights_from_checkpoint(self.model, checkpoint_config.path, model_name='model')
        self.completion_length = completion_length


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

        def normalize_text(text):
            return re.sub(r'[^\w]', '', text.lower())

        results = []
        for i, gen_tokens in enumerate(generated):
            decoded = self.tokenizer.decode(gen_tokens[-self.completion_length:], skip_special_tokens=True)

            answer_match = re.search(r'answer is (\d)', decoded, re.DOTALL)
            extracted_answer = answer_match.group(1).strip() if answer_match else ""
            expected_output = batch['expected_output'][i]
            normalized_extracted = normalize_text(extracted_answer)
            normalized_expected = normalize_text(expected_output)
            is_correct = normalized_extracted == normalized_expected

            results.append({
                'generated_text': decoded,
                'extracted_answer': extracted_answer,
                'expected_output': expected_output,
                'is_correct': is_correct
            })


        accuracy = sum(result['is_correct'] for result in results) / len(results)
        self.log("test/accuracy", accuracy, on_epoch=True, prog_bar=True)

        return results

    def test_dataloader(self):
        url = 'https://raw.githubusercontent.com/allenai/natural-instructions/refs/heads/master/tasks/task1333_check_validity_date_ddmmyyyy.json'
        response = requests.get(url)
        data = response.json()

        dataset = NaturalInstructionsDataset(data, self.tokenizer)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)

    def configure_optimizers(self):
        pass
