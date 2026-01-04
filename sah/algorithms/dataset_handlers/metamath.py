import re

import torch
from datasets import load_dataset

from .base import BaseDatasetHandler, GenerationValDataset, ProcessedTrainDataset


class MetaMathHandler(BaseDatasetHandler):
    def format_example(self, example):
        query = example['query']
        answer = example['response']
        return {
            "question": f"Question: {query}\nResponse:",
            "answer": f" {answer}"
        }

    def get_train_dataset(self):
        return ProcessedTrainDataset(
            self.tokenizer,
            self.dataset_name,
            self.format_example,
            self.block_size,
            max_examples=self.max_examples
        )

    def get_val_dataset(self):
        raw_dataset = load_dataset("gsm8k", "main", split="test")
        prompts = []
        answers = []

        for item in raw_dataset:
            question = item['question']
            answer = item['answer']

            answer_match = re.search(r'#### ([\d,]+)', answer)
            numerical_answer = answer_match.group(1).replace(',', '') if answer_match else ""

            prompt = f"Question: {question}\nResponse:"
            prompts.append(prompt)
            answers.append(numerical_answer)

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.block_size,
            padding_side='left',
        )

        return GenerationValDataset(tokenized['input_ids'], tokenized['attention_mask'], answers)

    def get_raw_val_data(self):
        if self.validation_data is None:
            raw_dataset = load_dataset("gsm8k", "main", split="test")
            prompts = []
            answers = []

            for item in raw_dataset:
                question = item['question']
                answer = item['answer']

                answer_match = re.search(r'#### ([\d,]+)', answer)
                numerical_answer = answer_match.group(1).replace(',', '') if answer_match else ""

                prompt = f"Question: {question}\nResponse:"
                prompts.append(prompt)
                answers.append(numerical_answer)

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
                max_new_tokens=pl_module.max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
        )

        correct_count = 0
        total_count = 0
        decoding_starts = batch['input_ids'].shape[1]

        for i, gen_tokens in enumerate(generated):
            decoded = self.tokenizer.decode(gen_tokens[decoding_starts:], skip_special_tokens=True)

            pattern = r'answer is:? ([\d,]+)'
            extracted_answer = ""
            match = re.search(pattern, decoded, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).replace(',', '')
            else:
                question_match = re.search(r'question:', decoded, re.IGNORECASE)
                if question_match:
                    text_before_question = decoded[:question_match.start()]
                    numbers = re.findall(r'[\d,]+', text_before_question)
                    if numbers:
                        extracted_answer = numbers[-1].replace(',', '')

            expected_answer = batch['expected_answer'][i]
            is_correct = extracted_answer == expected_answer

            if is_correct:
                correct_count += 1
            total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return {"performance": float(accuracy), "correct_count": float(correct_count), "total_count": float(total_count)}
