import io
import json
import os
import zipfile

import torch
from transformers import GPT2Tokenizer

from sah.algorithms.data_strategies.hashed_ngram import get_hashed_ngram


def load_jsonl(path):
    """Load data from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_zip_wikitext2(path):
    """Load wikitext-2 data from zip file."""
    data = []
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            if 'train' in member:
                with zf.open(member) as fh:
                    buf = []
                    for raw in io.TextIOWrapper(fh, encoding="utf-8"):
                        line = raw.rstrip("\n")
                        if line.startswith(' ='):          # blank line = new article
                            if len(buf) > 1:
                                data.append({"text": "\n".join(buf)})
                                buf.clear()
                        buf.append(line)
                    if buf:                      # flush last article
                        data.append({"text": "\n".join(buf)})
    return data


def tokenize_data(data, tokenizer, max_length=1024):
    """Tokenize text data and return sequences with masks."""
    sequences = []
    for item in data:
        text = item.get('text', '')
        if text.strip():
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            if len(tokens) > 0:
                # Create mask (all True for now, can be modified later)
                mask = [True] * len(tokens)
                sequences.append((tokens, mask))
    return sequences


def main():
    # Initialize GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Set paths
    dsir_output_dir = '/network/scratch/b/brownet/tmp'
    random_sample_path = '/network/scratch/b/brownet/tmp/random.jsonl'
    heuristic_sample_path = '/network/scratch/b/brownet/tmp/heuristic.jsonl'
    wikitext2_path = '/network/datasets/wikitext/wikitext-2-raw-v1.zip'

    print("Loading datasets...")

    # Load DSIR output (should be in jsonl format in the output directory)
    dsir_files = [f for f in os.listdir(dsir_output_dir) if f.endswith('.jsonl') and 'random' not in f]
    dsir_data = []
    for file in dsir_files:
        file_path = os.path.join(dsir_output_dir, file)
        dsir_data.extend(load_jsonl(file_path))

    # Load random sample
    random_data = load_jsonl(random_sample_path)

    # Load heuristic sample
    heuristic_data = load_jsonl(heuristic_sample_path)

    # Load wikitext-2 reference data
    wikitext2_data = load_zip_wikitext2(wikitext2_path)

    print(f"DSIR data: {len(dsir_data)} samples")
    print(f"Random data: {len(random_data)} samples")
    print(f"Heuristic data: {len(heuristic_data)} samples")
    print(f"WikiText-2 data: {len(wikitext2_data)} samples")

    print("Tokenizing datasets...")

    # Tokenize all datasets
    dsir_sequences = tokenize_data(dsir_data, tokenizer)
    random_sequences = tokenize_data(random_data, tokenizer)
    heuristic_sequences = tokenize_data(heuristic_data, tokenizer)
    wikitext2_sequences = tokenize_data(wikitext2_data, tokenizer)

    print(f"DSIR sequences: {len(dsir_sequences)}")
    print(f"Random sequences: {len(random_sequences)}")
    print(f"Heuristic sequences: {len(heuristic_sequences)}")
    print(f"WikiText-2 sequences: {len(wikitext2_sequences)}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters for hashed n-gram
    n = 4  # 4-gram
    entries = 2000
    outputs = tokenizer.vocab_size

    print("Creating hashed n-grams...")
    output_dir = '/network/scratch/b/brownet/tmp/ngrams'
    os.makedirs(output_dir, exist_ok=True)

    # Create hashed n-grams for each dataset
    dsir_ngrams = get_hashed_ngram(
        sequences=dsir_sequences,
        n=n,
        entries=entries,
        outputs=outputs,
        device=device,
        vocab_size=tokenizer.vocab_size
    )
    torch.save(dsir_ngrams, os.path.join(output_dir, 'dsir_ngrams.pt'))
    print(f"DSIR n-grams shape: {dsir_ngrams.shape}")

    random_ngrams = get_hashed_ngram(
        sequences=random_sequences,
        n=n,
        entries=entries,
        outputs=outputs,
        device=device,
        vocab_size=tokenizer.vocab_size
    )
    torch.save(random_ngrams, os.path.join(output_dir, 'random_ngrams.pt'))
    print(f"Random n-grams shape: {random_ngrams.shape}")

    heuristic_ngrams = get_hashed_ngram(
        sequences=heuristic_sequences,
        n=n,
        entries=entries,
        outputs=outputs,
        device=device,
        vocab_size=tokenizer.vocab_size
    )
    torch.save(heuristic_ngrams, os.path.join(output_dir, 'heuristic_ngrams.pt'))
    print(f"Heuristic n-grams shape: {heuristic_ngrams.shape}")

    wikitext2_ngrams = get_hashed_ngram(
        sequences=wikitext2_sequences,
        n=n,
        entries=entries,
        outputs=outputs,
        device=device,
        vocab_size=tokenizer.vocab_size
    )

    torch.save(wikitext2_ngrams, os.path.join(output_dir, 'wikitext2_ngrams.pt'))
    print(f"WikiText-2 n-grams shape: {wikitext2_ngrams.shape}")

    # Save n-grams
    print(f"N-grams saved to: {output_dir}")

    # Compute metric from proposed_strategy.py
    print("\nComputing proposed strategy metric:")

    # DSIR vs WikiText-2
    dsir_metric = compute_metric(dsir_ngrams.float(), wikitext2_ngrams.float())
    print(f"DSIR vs WikiText-2 metric: {dsir_metric.item():.6f}")

    # Random vs WikiText-2
    random_metric = compute_metric(random_ngrams.float(), wikitext2_ngrams.float())
    print(f"Random vs WikiText-2 metric: {random_metric.item():.6f}")

    # Heuristic vs WikiText-2
    heuristic_metric = compute_metric(heuristic_ngrams.float(), wikitext2_ngrams.float())
    print(f"Heuristic vs WikiText-2 metric: {heuristic_metric.item():.6f}")

    # Comparisons
    print(f"Difference (DSIR - Random): {(dsir_metric - random_metric).item():.6f}")
    print(f"Difference (DSIR - Heuristic): {(dsir_metric - heuristic_metric).item():.6f}")
    print(f"Difference (Heuristic - Random): {(heuristic_metric - random_metric).item():.6f}")


def compute_metric(
    current_ngram: torch.Tensor,   # (E, O)   counts  – "model"
    ngram:         torch.Tensor,   # (E, O)   counts  – "reference"
    alpha: float = 1.0,            # balances the terms
    eps: float = 1e-9,             # additive smoothing to avoid log(0)
):
    freq = ngram.to(torch.float32).sum(dim=-1)
    freq /= freq.sum()

    Q = current_ngram + eps
    P = ngram.to(torch.float32)       + eps

    Q = Q / Q.sum(dim=-1, keepdim=True)             # (E, O)
    P = P / P.sum(dim=-1, keepdim=True)

    H_Q = -(Q * Q.log()).sum(dim=-1)                # (E,)
    H_PQ = -(P * Q.log()).sum(dim=-1)               # (E,)

    return ((H_PQ -  alpha * H_Q) * freq).sum()


if __name__ == "__main__":
    main()
