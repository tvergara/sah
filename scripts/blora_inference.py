import math
import sys

import torch
import transformers as tr

t = " ".join(sys.argv[1:])
mn = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tk = tr.AutoTokenizer.from_pretrained(mn)
m = tr.AutoModelForCausalLM.from_pretrained(mn)
ofs = math.log(0.1 / 1.1) * 2 / 3

def q(x, s2, gm):
    g = [(torch.sigmoid(ofs - gm[i]) < 0.34).float() for i in range(4)]
    s = [s2]
    [s.append(s[-1] / d) for d in [5, 17, 257, 65537]]
    xq = s[0] * torch.round(x / s[0])
    c = 1.0
    for i in range(4):
        c = c * g[i]
        xq = xq + c * s[i + 1] * torch.round((x - xq) / s[i + 1])
    return xq

w = torch.load("adapter.pt")
for k, v in w.items():
    A, B, E = q(v["A"], v["sA"], v["gA"]), q(v["B"], v["sB"], v["gB"]), q(v["E"], v["sE"], v["gE"])
    g2 = torch.cumprod((torch.sigmoid(ofs - v["g2"]) < 0.34).float(), 0)
    E = torch.cat([torch.ones(1, 1), g2.unsqueeze(1)]) * E
    mod = m.get_submodule(k)
    mod.weight.data += (B @ (A * E)) * v["sc"]

i = tk(t, return_tensors="pt")
o = m.generate(**i, max_new_tokens=100)
print(tk.decode(o[0]))
