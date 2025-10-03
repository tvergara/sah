import re

import torch
from transformers import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from sah.algorithms.utils import load_weights_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_chat = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16).eval().to(device)


load_weights_from_checkpoint(model_chat, '/network/scratch/b/brownet/hydra-runs/finetune-on-meta_math/checkpoints/last', model_name='model')


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen2.5-1.5B"
)

prompt = "Question: Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?\nResponse:"

tokenized = tokenizer(
    prompt,
    truncation=True,
    padding=False,
    max_length=512,
    return_tensors="pt"
)



# Generate text based on the prompt
with torch.no_grad():
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    # Generate text
    generated = model_chat.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_new_tokens=300,  # Adjust as needed
        max_new_tokens=300,  # Adjust as needed
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)
    answer_match = re.search(r'answer is (\d)', generated_text, re.DOTALL)
    extracted_answer = answer_match.group(1).strip() if answer_match else ""
    extracted_answer
