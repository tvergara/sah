from decimal import Decimal, getcontext

import torch

getcontext().prec = 2000


def normalize_pdf(probs):
    eps = torch.finfo(probs.dtype).eps
    vocab_size = probs.shape[0]
    normalized = (1 - 2 * vocab_size * eps) * probs + eps
    return normalized / normalized.sum()


def encode_single(tokens, probs):
    low, high = Decimal(0), Decimal(1)

    for i, token in enumerate(tokens[1:], start=0):
        token_probs = normalize_pdf(probs[i].cpu())
        cumprobs = torch.cumsum(token_probs, dim=0)

        range_size = high - low
        prev_cum = Decimal(0) if token == 0 else Decimal(str(cumprobs[token - 1].item()))
        curr_cum = Decimal(str(cumprobs[token].item()))

        high = low + range_size * curr_cum
        low = low + range_size * prev_cum

    return (low + high) / 2, high - low


def decode_single(compressed, probs, num_tokens):
    low, high = Decimal(0), Decimal(1)
    code = compressed
    decoded = []

    for i in range(num_tokens - 1):
        token_probs = normalize_pdf(probs[i].cpu())
        cumprobs = torch.cumsum(token_probs, dim=0)

        range_size = high - low
        scaled = (code - low) / range_size

        cumprobs_list = [Decimal(str(p.item())) for p in cumprobs]
        token = 0
        for idx, cp in enumerate(cumprobs_list):
            if scaled < cp:
                token = idx
                break

        decoded.append(token)

        prev_cum = Decimal(0) if token == 0 else Decimal(str(cumprobs[token - 1].item()))
        curr_cum = Decimal(str(cumprobs[token].item()))

        high = low + range_size * curr_cum
        low = low + range_size * prev_cum

    return decoded


def encode(model, input_ids_batch, attention_mask_batch=None):
    if isinstance(input_ids_batch, list):
        input_ids_batch = torch.tensor(input_ids_batch)

    if input_ids_batch.dim() == 1:
        input_ids_batch = input_ids_batch.unsqueeze(0)

    input_ids_batch = input_ids_batch.to(model.device)

    if attention_mask_batch is not None:
        if isinstance(attention_mask_batch, list):
            attention_mask_batch = torch.tensor(attention_mask_batch)
        if attention_mask_batch.dim() == 1:
            attention_mask_batch = attention_mask_batch.unsqueeze(0)
        attention_mask_batch = attention_mask_batch.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids_batch)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    compressed_batch = []
    for batch_idx in range(input_ids_batch.shape[0]):
        tokens = input_ids_batch[batch_idx].cpu().tolist()

        if attention_mask_batch is not None:
            attention_mask = attention_mask_batch[batch_idx].cpu().tolist()
            actual_length = sum(attention_mask)
            tokens = tokens[:actual_length]
            sequence_probs = probs[batch_idx, :actual_length]
        else:
            sequence_probs = probs[batch_idx]

        compressed_value, interval_width = encode_single(tokens, sequence_probs)
        compressed_batch.append((compressed_value, interval_width))

    return compressed_batch


def decode(model, compressed_batch, num_tokens_list):
    decoded_batch = []

    for (compressed_value, _), num_tokens in zip(compressed_batch, num_tokens_list):
        first_token_id = 0
        input_ids = torch.tensor([[first_token_id]]).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        full_probs_list = [probs[0, 0]]

        for _ in range(num_tokens - 1):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                full_probs_list.append(probs[0, -1])

        probs_tensor = torch.stack(full_probs_list)
        decoded_tokens = decode_single(compressed_value, probs_tensor, num_tokens)
        decoded_batch.append([first_token_id] + decoded_tokens)

    return decoded_batch
