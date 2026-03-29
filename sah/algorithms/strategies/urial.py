import os

import torch
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.arithmetic_coding import compute_bits_from_logits

SCRATCH_DIR = os.environ.get('SCRATCH', '/network/scratch/b/brownet')

DATASET_PROMPTS = {
    "meta-math/MetaMathQA": """# System
You are a helpful, respectful and honest assistant.

# Instruction
Solve the following math problem step by step. Show your reasoning clearly.

Question: Sue works in a factory and every 30 minutes, a machine she oversees produces 30 cans of soda. How many cans of soda can x machine produce in 8 hours? If we know the answer to the above question is 480, what is the value of unknown variable x?

Response: We know that every 30 minutes, a machine produces 30 cans of soda.
Since there are 60 minutes in an hour, and 8 hours in total, the total number of minutes is 60 * 8 = 480 minutes.
If a machine produces 30 cans of soda every 30 minutes, then in 480 minutes, it will produce (480/30) * 30 = 480 cans of soda.
We are given that the total number of cans of soda produced is 480, so we can write: 480 = 480 * x.
Dividing both sides by 480, we get: x = 1.
The value of x is 1.
#### 1
The answer is: 1

Question: Maximoff's monthly bill is $60 per month. His monthly bill increased by thirty percent when he started working at home. How much is his total monthly bill working from home?

Response: Maximoff's monthly bill increased by 30% when he started working from home, so his bill increased by 30/100 * $60 = $18
His new monthly bill working from home is $60 + $18 = $78
#### 78
The answer is: 78

""",
    "allenai/nllb": """# System
You are a helpful assistant.

# Instruction
Translate from english to french.

English: They also reduce appetite (66).

French: Ils r\u00e9duisent \u00e9galement l'app\u00e9tit (66).

English: For Bauman, modern humans need strong commitments.

French: Pour Bauman, les humains modernes ont besoin d\u2019engagements forts.
""",
    f"ifeval:{SCRATCH_DIR}/correct_ifeval_examples.jsonl": """<|im_start|>system
You are a helpful, respectful and honest assistant. You carefully follow instructions to every detail possible.

<|im_start|>user
Describe the purpose of internet protocol version 6 (IPv6).

Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.
<|im_start|>assistant
ipv6 is designed to replace ipv4 as the internet\u2019s addressing system. it provides a much larger pool of unique addresses to accommodate the growing number of devices connected to the internet. additionally, ipv6 improves efficiency by reducing the size of header fields and enabling simpler routing. it also includes features like built-in security extensions and autoconfiguration to simplify network administration.
<|im_start|>user
Your entire response should be in English, capital letters only. Which country has the largest population in the world?
<|im_start|>assistant
CHINA
""",
}
DATASET_PROMPTS["esnli"] = """<|im_start|>system
You are a helpful, respectful and honest assistant. Given a premise and a hypothesis, explain your reasoning and then classify the relationship as entailment, neutral, or contradiction.

<|im_start|>user
Premise: A person on a horse jumps over a broken down airplane.
Hypothesis: A person is outdoors, on a horse.
Does the premise entail, contradict, or is neutral to the hypothesis? Explain your reasoning, then answer.
<|im_start|>assistant
A broken down airplane is outdoors, and the person is on a horse jumping over it. Therefore, the answer is entailment.
<|im_start|>user
Premise: Children smiling and waving at camera
Hypothesis: The kids are frowning
Does the premise entail, contradict, or is neutral to the hypothesis? Explain your reasoning, then answer.
<|im_start|>assistant
One cannot be smiling and frowning at the same time. Therefore, the answer is contradiction.
"""
DATASET_PROMPTS[f"ifeval:{SCRATCH_DIR}/correct_ifeval_examples_extended_32.jsonl"] = DATASET_PROMPTS[f"ifeval:{SCRATCH_DIR}/correct_ifeval_examples.jsonl"]
DATASET_PROMPTS[f"ifeval:{SCRATCH_DIR}/correct_ifeval_examples_extended_32_clean.jsonl"] = DATASET_PROMPTS[f"ifeval:{SCRATCH_DIR}/correct_ifeval_examples.jsonl"]

class URIALStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)

        dataset_name = pl_module.dataset_name
        self.prompt = DATASET_PROMPTS[dataset_name]

    def on_validation_start(self, pl_module):
        prompt_tokens = pl_module.tokenizer.encode(self.prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_tokens], device=pl_module.model.device)

        with torch.no_grad():
            outputs = pl_module.model(input_ids)
            logits = outputs.logits

        self.bits = compute_bits_from_logits(logits, input_ids, attention_mask=None, fast=True)

    def validation_step(self, pl_module, batch, batch_idx):
        questions = batch["question"]
        full_prompts = [self.prompt + q for q in questions]

        tokenized = pl_module.tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )
        input_ids = tokenized["input_ids"].to(pl_module.device)
        attention_mask = tokenized["attention_mask"].to(pl_module.device)

        prepared_batch = {
            **batch,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        metrics = self.dataset_handler.validate_batch(pl_module, prepared_batch, batch_idx)
        batch_size = len(questions)

        for key, value in metrics.items():
            if key in ["correct_count", "total_count"]:
                pl_module.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum", batch_size=batch_size)
            else:
                pl_module.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def val_dataloader(self, pl_module):
        raw_val_data = self.dataset_handler.get_raw_val_data()
        dataset = URIALValidationDataset(raw_val_data)
        return DataLoader(
            dataset,
            batch_size=pl_module.val_batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def train_dataloader(self, pl_module):
        return None

    def configure_optimizers(self, pl_module):
        return None


class URIALValidationDataset(Dataset):
    def __init__(self, raw_val_data):
        self.data = raw_val_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
