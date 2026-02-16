import pickle
import sys

import gptzip
import torch
import transformers as tr

p = sys.argv[2]
mn = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tk = tr.AutoTokenizer.from_pretrained(mn)
tk.pad_token = tk.eos_token
m = tr.AutoModelForCausalLM.from_pretrained(mn)
m.train()
cd = gptzip.ArithmeticCoder(lm=m, tokenizer=tk)
op = torch.optim.AdamW(m.parameters(), lr=1e-5)

for line in open(sys.argv[1]):
    with open(line.strip(), "rb") as f:
        c, n = pickle.load(f)
    i = tk(cd.decode(c, num_padded_bits=n), return_tensors="pt")
    op.zero_grad()
    m(**i, labels=i["input_ids"].clone()).loss.backward()
    op.step()

m.eval()
print(tk.decode(m.generate(**tk(p, return_tensors="pt"), max_new_tokens=100)[0]))
