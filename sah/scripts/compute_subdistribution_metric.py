from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import OmegaConf


# ─── Hydra config ────────────────────────────────────────────────────────────
@dataclass
class AutomataCfg:
    variant_path: str = "modified-automaton"   # B
    original_path: str = "initial-automaton"   # A (baseline)

@dataclass
class EvalCfg:
    device: str = "cuda"
    eps: float = 1e-12

# NEW ▶▶ where to store results + run identifier
@dataclass
class ResultCfg:
    file: str = "results.csv"   # set to "" to disable writing
    run_id: str = "exp000"      # unique identifier for this run

@dataclass
class Config:
    automaton: AutomataCfg = field(default_factory=AutomataCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    result: ResultCfg = field(default_factory=ResultCfg)


# ─── tiny helpers ────────────────────────────────────────────────────────────

def pad_stack(seqs: list[list[int]], pad: int = -1) -> torch.Tensor:
    L = max(map(len, seqs))
    t = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return t


def load_pfsa(path: Path):
    d = yaml.safe_load((path / "automaton.yaml").read_text())
    vocab, tp, nc = d["vocab"], d["token_probs"], d["next_state_cond"]
    return vocab, torch.as_tensor(tp, dtype=torch.float64), torch.as_tensor(nc, dtype=torch.float64)


def load_ids(path: Path, vocab: list[str]):
    tok2id = {t: i for i, t in enumerate(vocab)}
    seqs = [[tok2id[t] for t in ln.split()] for ln in (path / "test.txt").read_text().splitlines()]
    return pad_stack(seqs)  # (B, L)


# ─── core computation ────────────────────────────────────────────────────────
@torch.no_grad()
def entropy_and_cross(
    ids: torch.Tensor, E_A, T_A, E_B, T_B,
    device="cuda", eps=1e-12
):
    PAD = -1
    ids, E_A, T_A, E_B, T_B = (x.to(device) for x in (ids, E_A, T_A, E_B, T_B))
    mask = ids.ne(PAD)
    B, L = ids.shape
    S, V = E_A.shape
    T_Ay, T_By = T_A.permute(1,0,2), T_B.permute(1,0,2)  # (V, S, S)

    # beliefs
    bA = torch.zeros(B, S, dtype=torch.float64, device=device)
    bA[:,0] = 1
    bB = torch.zeros_like(bA)
    bB[:,0] = 1

    ent_A = cross_BA = tot = 0.0
    for t in range(L):
        act = mask[:, t]
        if not act.any():
            break
        y = ids[act, t]

        # predictive dists
        p_A = bA[act] @ E_A          # (B_act, V)
        p_B = bB[act] @ E_B          # (B_act, V)

        # metrics
        ent_A   += (-(p_A * p_A.clamp_min(eps).log()).sum(dim=1)).sum().item()
        cross_BA+= (-(p_B * p_A.clamp_min(eps).log()).sum(dim=1)).sum().item()
        tot     += p_A.size(0)

        # forward update A
        tmpA = bA[act] * E_A.t()[y]
        bA_next = torch.bmm(tmpA.unsqueeze(1), T_Ay[y]).squeeze(1)
        bA_next /= bA_next.sum(dim=1, keepdim=True)
        bA[act] = bA_next

        # forward update B
        tmpB = bB[act] * E_B.t()[y]
        bB_next = torch.bmm(tmpB.unsqueeze(1), T_By[y]).squeeze(1)
        bB_next /= bB_next.sum(dim=1, keepdim=True)
        bB[act] = bB_next

    return ent_A / tot, cross_BA / tot


# ─── pipeline ────────────────────────────────────────────────────────────────

def run(cfg: Config):
    var_dir, ori_dir = Path(cfg.automaton.variant_path), Path(cfg.automaton.original_path)

    vocab_B, E_B, T_B = load_pfsa(var_dir)
    vocab_A, E_A, T_A = load_pfsa(ori_dir)

    if vocab_A != vocab_B:
        raise ValueError("Vocabularies differ; align them before comparing.")

    ids = load_ids(var_dir, vocab_B)

    H_A, H_BA = entropy_and_cross(ids, E_A, T_A, E_B, T_B,
                                  device=cfg.eval.device, eps=cfg.eval.eps)

    # console report
    print(f"\nAverage predictive entropy  H(A)            : {H_A:.4f} nats "
          f"({H_A/math.log(2):.4f} bits)")
    print(f"Average cross-entropy      H(B, A) = H_BA  : {H_BA:.4f} nats "
          f"({H_BA/math.log(2):.4f} bits)")
    print(f"Δ (H_BA − H_A)                                : {H_BA-H_A:+.4f} nats "
          f"({(H_BA-H_A)/math.log(2):+.4f} bits)")

    # ── NEW: persist to disk ────────────────────────────────────────────────
    if cfg.result.file:
        out_path = Path(cfg.result.file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["id", "H_A", "H_BA"])
            writer.writerow([cfg.result.run_id, H_A, H_BA])
        print(f"Results appended to {out_path.resolve()}")


# ─── CLI entrypoint ──────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path=None)
def main(cfg: Config):  # type: ignore[arg-type]
    print("Hydra cfg:\n" + OmegaConf.to_yaml(cfg))
    run(cfg)

if __name__ == "__main__":
    main()
