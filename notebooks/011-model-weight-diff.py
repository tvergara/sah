#!/usr/bin/env python3

from collections import OrderedDict

import tensorly as tl
import torch
from tensorly.decomposition import tucker
from transformers import AutoModel


def download_and_load_models():
    """Download and load both models."""
    print("Loading huggingface-course/bert-finetuned-ner...")
    ner_model = AutoModel.from_pretrained("huggingface-course/bert-finetuned-ner")

    print("Loading google-bert/bert-base-cased...")
    base_model = AutoModel.from_pretrained("google-bert/bert-base-cased")

    return ner_model, base_model

def compute_weight_diff(ner_model, base_model):
    """Compute the difference in weights between fine-tuned and base models."""
    ner_state_dict = ner_model.state_dict()
    base_state_dict = base_model.state_dict()

    weight_diff = OrderedDict()

    print("\nComputing weight differences...")
    for key in ner_state_dict.keys():
        if key in base_state_dict:
            diff = ner_state_dict[key] - base_state_dict[key]
            weight_diff[key] = diff

            # Print statistics about the difference
            diff_norm = torch.norm(diff).item()
            original_norm = torch.norm(base_state_dict[key]).item()
            relative_change = diff_norm / original_norm if original_norm > 0 else 0

            print(f"{key}: diff_norm={diff_norm:.6f}, relative_change={relative_change:.6f}")
        else:
            print(f"Warning: {key} not found in base model")

    return weight_diff

def analyze_weight_diff(weight_diff):
    """Analyze the weight differences."""
    print("\n=== Weight Difference Analysis ===")

    total_params = 0
    total_diff_norm = 0
    layer_stats = {}

    for key, diff in weight_diff.items():
        param_count = diff.numel()
        diff_norm = torch.norm(diff).item()

        total_params += param_count
        total_diff_norm += diff_norm

        # Group by layer type
        layer_type = key.split('.')[0] if '.' in key else key
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {'count': 0, 'norm': 0, 'params': 0}

        layer_stats[layer_type]['count'] += 1
        layer_stats[layer_type]['norm'] += diff_norm
        layer_stats[layer_type]['params'] += param_count

    print(f"Total parameters with differences: {total_params:,}")
    print(f"Total difference norm: {total_diff_norm:.6f}")
    print(f"Average difference norm: {total_diff_norm / len(weight_diff):.6f}")

    print("\nBy layer type:")
    for layer_type, stats in layer_stats.items():
        avg_norm = stats['norm'] / stats['count']
        print(f"  {layer_type}: {stats['count']} layers, avg_norm={avg_norm:.6f}, params={stats['params']:,}")

def create_diff_model(base_model, weight_diff):
    """Create a new model that represents the weight difference."""
    print("\n=== Creating Difference Model ===")

    # Clone the base model architecture
    diff_model = type(base_model)(base_model.config)

    # Set the weights to the differences
    diff_state_dict = OrderedDict()
    for key, diff in weight_diff.items():
        diff_state_dict[key] = diff

    diff_model.load_state_dict(diff_state_dict, strict=False)

    print(f"Created difference model with {sum(p.numel() for p in diff_model.parameters()):,} parameters")

    return diff_model

def main():
    print("=== Model Weight Difference Experiment ===")

    # Download and load models
    ner_model, base_model = download_and_load_models()

    # Compute weight differences
    weight_diff = compute_weight_diff(ner_model, base_model)

    # Analyze the differences
    analyze_weight_diff(weight_diff)

    # Create difference model
    diff_model = create_diff_model(base_model, weight_diff)

    print("\n=== Experiment Complete ===")
    print("Models loaded:")
    print("- ner_model: fine-tuned NER model")
    print("- base_model: base BERT model")
    print("- diff_model: model containing weight differences")
    print("- weight_diff: dictionary of weight differences")

    return ner_model, base_model, diff_model, weight_diff

if __name__ == "__main__":
    ner_model, base_model, diff_model, weight_diff = main()

    layers = len(diff_model.encoder.layer)
    queries = []
    for i in range(layers):
        queries.append(diff_model.encoder.layer[0].output.dense.weight)

    full_query = torch.stack(queries)
    full_query.shape

    tl.set_backend('pytorch')
    tucker_tensor = tucker(full_query, rank=[12, 40, 40])

    tucker_tensor.factors[0].shape
    tucker_tensor.factors[1].shape
    tucker_tensor.factors[2].shape
    len(tucker_tensor.factors)

    tucker_tensor.core.shape

    torch.numel(tucker_tensor.core)

    total = sum(torch.numel(tucker_tensor.factors[i]) for i in range(3)) + torch.numel(tucker_tensor.core)
    total
    total / torch.numel(full_query)

    reconstructed = tl.tucker_to_tensor(tucker_tensor)

    torch.norm(reconstructed - full_query) / torch.norm(full_query)
    full_query
    reconstructed
