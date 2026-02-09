import pickle
import sys

import transformers as tr

t = " ".join(sys.argv[1:])
mn = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tk = tr.AutoTokenizer.from_pretrained(mn)
m = tr.AutoModelForCausalLM.from_pretrained(mn)

with open("code.pkl", "rb") as f:
    c, n = pickle.load(f)

i = tk(t, return_tensors="pt")
o = m.generate(**i, max_new_tokens=100)
print(tk.decode(o[0]))
