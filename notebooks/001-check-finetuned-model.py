from transformers.models.auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from sah.algorithms.utils import load_weights_from_checkpoint

pretrained_model_name_or_path = 'HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints'
revision = 'step-160000'

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, revision=revision)

model.eval()

model

path = "/home/mila/b/brownet/scratch/hydra-runs/alpaca/checkpoints/step_00050"
load_weights_from_checkpoint(model, path)

tokenizer = AutoTokenizer.from_pretrained(
    'HuggingFaceTB/SmolLM2-360M',
    use_fast= True,
    trust_remote_code= True
)

model


prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Give three tips for staying healthy.

### Response:"""
inputs = tokenizer(prompt, return_tensors="pt")


generated_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)


print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
