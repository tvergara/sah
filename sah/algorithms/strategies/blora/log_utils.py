import numpy as np
import torch
import torch.nn as nn

import wandb

from .straight_through import Quantizer


def print_and_log(*s, logfile=None, **kwargs):
    print(*s, **kwargs)
    if logfile:
        print(*s, **kwargs, file=open(logfile, "a"))


def get_bw(quantizer, prune_only=False):
    if prune_only:
        if quantizer.quantizer is None or not (
            quantizer.quantizer.fixed_8bit or quantizer.quantizer.fixed_4bit
        ):
            return 16
    if "bayesian_bits" in quantizer.method:
        assert quantizer.quantizer is not None
        if quantizer.quantizer.fixed_8bit:
            return 8
        elif quantizer.quantizer.fixed_4bit:
            return 4

        train = quantizer.quantizer.training
        if train:
            quantizer.quantizer.eval()

        fix_type = lambda g: int(g.item()) if isinstance(g, torch.Tensor) else int(g)
        q4, q8, q16, q32 = (fix_type(g) for g in quantizer.quantizer.get_gates()[1:])
        n = 1 + q4 + q4 * (q8 + q8 * (q16 + q16 * q32))

        if train:
            quantizer.quantizer.train()

        return int(2**n)
    else:
        return quantizer.n_bits


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_relevant_quantizers_from_groups(quantizer_groups):
    result = []
    for qg, quantizers in quantizer_groups:
        result += list(zip(qg, quantizers))
    return result


def get_quantizers(model: nn.Module):
    quantizers: list[tuple[str, Quantizer]] = []
    for n, m in model.named_modules():
        if isinstance(m, Quantizer):
            quantizers.append((n, m))

    return quantizers


def pretty_print_quantization(
    module,
    method,
    logfile=None,
    prune_only=False,
    log_wandb=False,
    prune_rank=False,
    quantize=True,
    max_rank=0,
):
    relevant_quantizers = get_quantizers(module)
    apply_sigmoid = method == "bayesian_bits"
    n_bits_txt = ""
    max_len = max([len(nm) for nm, _ in relevant_quantizers])

    template = "| {{:<{ml}}}|"

    if quantize:
        template += (
            " {{:<7}} || "
            "{{:>{ns}.4f}} | {{:>{ns}.4f}} | {{:>{ns}.4f}} "
            "| {{:>{ns}.4f}} |"
        )

        template += "| {{x_min:>8.4f}} | {{x_max:>8.4f}} |"

    if prune_rank:
        template += "| {{:>{ns}}} |"
        rank_template = " | ".join(["{{:>{ns}.4f}}" for _ in range(max_rank)]) + "\n"
        rank_template = rank_template.format(ns=16)
        rank_txt = ""

    template += "\n"
    template = template.format(ml=max_len, ns=6 + 2 * int(not apply_sigmoid), lpi=8)

    dummy_zeros = [0] * 4 * (1 + (method == "bayesian_bits"))
    if prune_rank and quantize:
        dummy_line = template.format("a", 8, *dummy_zeros, "", x_min=0.0, x_max=0.0)
        dummy_rank_line = rank_template.format(*[0 for _ in range(max_rank)])
        rank_hline = "|" + "-" * (len(dummy_rank_line) - 3) + "|"
    elif quantize:
        dummy_line = template.format("a", 8, *dummy_zeros, x_min=0.0, x_max=0.0)
    else:
        dummy_line = template.format("a", 8)
        dummy_rank_line = rank_template.format(*[0 for _ in range(max_rank)])
        rank_hline = "|" + "-" * (len(dummy_rank_line) - 3) + "|"

    hline = "|" + "-" * (len(dummy_line) - 3) + "|"

    if log_wandb:
        wandb_log = {}

    for name, quantizer in relevant_quantizers:
        bw = get_bw(quantizer, prune_only)
        if not prune_only and method is not None:
            gams = [
                getattr(quantizer.quantizer, f"gamma_{2**i}").item()
                for i in range(2, 6)
            ]
        else:
            gams = [0] * 4
        if apply_sigmoid:
            gams = [sigmoid(g) for g in gams]
        if quantizer.quantizer is not None:
            x_min, x_max = (
                quantizer.quantizer.x_min.item(),
                quantizer.quantizer.x_max.item(),
            )
        else:
            x_min = x_max = float("nan")

        if prune_rank and quantizer.quantizer.prune_rank:
            rank = quantizer.quantizer.curr_rank
        else:
            rank = 0

        if prune_rank and quantize:
            n_bits_txt += template.format(
                name, int(np.log2(bw)) * "*", *gams, rank, x_min=x_min, x_max=x_max
            )
            if quantizer.quantizer.gamma_2 is not None:
                rank_gammas = [1.0] + [
                    sigmoid(g.item()) for g in quantizer.quantizer.gamma_2
                ]
                rank_txt += rank_template.format(*rank_gammas)
        elif quantize:
            n_bits_txt += template.format(
                name, int(np.log2(bw)) * "*", *gams, x_min=x_min, x_max=x_max
            )
        else:
            if rank > 0:
                n_bits_txt += template.format(
                    name,
                    rank,
                )
            if quantizer.quantizer.gamma_2 is not None:
                rank_gammas = [1.0] + [
                    sigmoid(g.item()) for g in quantizer.quantizer.gamma_2
                ]
                rank_txt += rank_template.format(*rank_gammas)

        if log_wandb:
            g = 4
            for i in range(len(gams)):
                wandb_log[f"train/{name}/gamma_{g}"] = gams[i]
                g *= 2

    print_and_log(hline.replace("|", "-"), logfile=logfile)

    # Make header: | Quantizer | ln(B) || g2 | ... | g32 | x_min | x_max |
    # make title elements for header, put in list:
    hs = ["g"]
    header = ["Quantizer", "log2(B)"] + [h + str(2**i) for h in hs for i in range(2, 6)]
    if prune_rank:
        header.append("Rank")
        rank_header = [f"Gamma Rank {i + 1}" for i in range(max_rank)]
    # format header with title elements:
    print_and_log(
        template.replace(".4f", "")
        .replace(".2f", "")
        .format(*header, x_min="x_min", x_max="x_max"),
        end="",
        logfile=logfile,
    )
    print_and_log(hline, logfile=logfile)

    # Add the rest of the text
    print_and_log(n_bits_txt, end="", logfile=logfile)
    print_and_log(hline.replace("|", "-"), logfile=logfile)

    if prune_rank and max_rank <= 8:
        print_and_log(
            rank_template.replace(".4f", "").replace(".2f", "").format(*rank_header),
            end="",
            logfile=logfile,
        )
        print_and_log(rank_hline, logfile=logfile)

        print_and_log(rank_txt, end="", logfile=logfile)
        print_and_log(rank_hline.replace("|", "-"), logfile=logfile)

    print_and_log(logfile=logfile)

    if log_wandb:
        return wandb_log


def print_bitops(
    per_layer_macs,
    quantizer_groups,
    logfile=None,
    no_print=False,
    prune_only=False,
    rank=0,
    log_wandb=False,
):
    # We assume input has 8bit color values per channel (even if it's FP32 encoded)
    total_bitops = 0
    n_act, n_weight = ([0] * 5), ([0] * 5)
    baseline = sum([v * (32**2) for n, v in per_layer_macs.items()])
    found_input = False

    for qg, quantizers in quantizer_groups:
        act_bw, weights = 0, []
        found_act = False
        if len(qg) == 1:
            found_act = found_input = True
            act_name, act_bw = "input", 32

        for name, quantizer in zip(qg, quantizers):
            if "act" in name or "p2c" in name or "c2p" in name or "scores" in name:
                # assert not found_act
                found_act = True
                act_name = name
                act_bw = get_bw(quantizer, prune_only)

            elif "weight" in name or "lora" in name:
                weight_bw = get_bw(quantizer, prune_only)
                # prune_ratio, _ = get_keep_ratio_and_prune_prob(
                #     quantizer, name, model, rn
                # )
                prune_ratio = 1.0

                weights.append((name, weight_bw, prune_ratio))

            else:
                raise ValueError("Unknown quantizer:", name)

        if act_name != "input":
            act_idx = int(np.log2(act_bw) - 1)
            n_act[act_idx] += 1

        for weight_name, weight_bw, prune_ratio in weights:
            if rank > 0:
                total_bitops += (
                    act_bw * weight_bw * prune_ratio * per_layer_macs[weight_name]
                )

            total_bitops += (
                act_bw * weight_bw * prune_ratio * per_layer_macs[weight_name]
            )
            weight_idx = int(np.log2(weight_bw) - 1)
            weight_idx = max(0, min(4, weight_idx))  # bw = 2 ** (weight_idx + 1)
            n_weight[weight_idx] += 1

    args = []
    if log_wandb:
        w_args, a_args = [], []
        n_bits = 2
    for i in range(len(n_weight)):
        args.append(n_weight[i])
        args.append(n_act[i])
        if log_wandb:
            w_args.append([n_bits, n_weight[i]])
            a_args.append([n_bits, n_act[i]])
            n_bits *= 2
    args += [total_bitops / baseline * 100]
    print("total:", total_bitops)
    nbits_summary_str = (
        "bits: weights:   activations:\n"
        "  2:  {:>5}      {:>5}\n"
        "  4:  {:>5}      {:>5}\n"
        "  8:  {:>5}      {:>5}\n"
        " 16:  {:>5}      {:>5}\n"
        " 32:  {:>5}      {:>5}\n"
        "relative bops: {:.2f}%".format(*args)
    )

    if log_wandb:
        table_w = wandb.Table(data=w_args, columns=["n_bits", "counts"])
        table_a = wandb.Table(data=a_args, columns=["n_bits", "counts"])

    if not no_print:
        print_and_log(nbits_summary_str, logfile=logfile)
        print_and_log(
            f"GBOPs: {total_bitops * 10 ** (-9):<6.2}, GBOPs baseline: {baseline * 10 ** (-9):<6.2}",
            logfile=logfile,
        )

    if log_wandb:
        return total_bitops / baseline * 100, nbits_summary_str, table_w, table_a
    return total_bitops / baseline * 100, nbits_summary_str, None, None
