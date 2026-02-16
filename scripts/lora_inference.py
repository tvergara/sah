import sys

import peft
import transformers as tr

t = " ".join(sys.argv[1:])
mn = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tk = tr.AutoTokenizer.from_pretrained(mn)
m = tr.AutoModelForCausalLM.from_pretrained(mn)
m = peft.PeftModel.from_pretrained(m, "adapter")
i = tk(t, return_tensors="pt")
o = m.generate(**i, max_new_tokens=100)
print(tk.decode(o[0]))
