import copy

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

from sah.algorithms.networks.transformer import Transformer
from sah.algorithms.utils import load_weights_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer_config = {
    'vocab_size':12,
    'd_model':50,
    'dim_feedforward':200,
    'n_heads':5,
    'n_layers':5
}

base_model = Transformer(**transformer_config).to(device)
pretrained_checkpoint = '/network/scratch/b/brownet/hydra-runs/grammar/checkpoints/pretraining_step_02600'
load_weights_from_checkpoint(base_model, pretrained_checkpoint, model_name='transformer')



finetuned_model = Transformer(**transformer_config).to(device)
finetuned_ckp = '/network/scratch/b/brownet/hydra-runs/finetune-grammar/checkpoints/finetuning_step_00020'
load_weights_from_checkpoint(finetuned_model, finetuned_ckp, model_name='transformer')


diffs = [p_fine - p_pre for p_fine, p_pre in zip(finetuned_model.parameters(), base_model.parameters())]

batches = 300
batch_size = 64
expansion = 5
total_data = batches * batch_size * expansion

seq_len = 100

vocab_size = base_model.tok_emb.num_embeddings

input_ids = torch.zeros((total_data, seq_len), dtype=torch.long, device=device)
input_ids[:, 0] = torch.randint(1, vocab_size-1, (total_data,), device=device)

total_labels = torch.zeros((total_data, seq_len, vocab_size), dtype=torch.float16, device=device)


for i in range(1, seq_len):
    for j in range(0, total_data, 200):
        with torch.no_grad():
            j_max = min(j + 200, total_data)
            outputs = finetuned_model(input_ids[j:j_max, :i])[:, -1]

            dist = torch.distributions.Categorical(logits=outputs)
            samples = dist.sample()
            input_ids[j:j_max, i] = samples

for i in range(0, total_data, 100):
    i_max = min(total_data, i + 100)
    x = input_ids[i:i_max]
    total_labels[i:i_max] = finetuned_model(x).detach().clone()


def compute_loss(y_hat, y_star):
    assert y_hat.shape == y_star.shape
    temperature = 1.0
    log_probs_student = F.log_softmax(y_hat / temperature, dim=-1)
    probs_teacher = F.softmax(y_star / temperature, dim=-1)
    kl_loss = F.kl_div(log_probs_student, probs_teacher, reduction="batchmean") * (temperature ** 2)
    return kl_loss

dataset = torch.randperm(total_data, device=device)[:batches * batch_size].view(batches, batch_size)
# dataset = torch.randint(0, total_data - 1, (batches, batch_size), device=device)
used = set(dataset.flatten().tolist())
LR = 0.0001


META_BATCHES = 100
SUBSPACE = 2048
SELECTION_BATCHES = 50
rand_mapping = [torch.rand(p.numel(), SUBSPACE, device=device) for p in base_model.parameters()]
query = torch.zeros(SUBSPACE, device=device)
with torch.no_grad():
    for param, mapping in zip(diffs, rand_mapping):
        query += param.view(-1) @ mapping  # [batch_size, subspace_size]

for _ in tqdm(range(META_BATCHES)):
    model = copy.deepcopy(base_model)
    db = torch.zeros(total_data, SUBSPACE, device=device)

    def update_db(model):
        for i in range(0, total_data, SELECTION_BATCHES):
            i_max = min(i + SELECTION_BATCHES, total_data)
            batch = input_ids[i:i_max]
            labels = total_labels[i:i_max]

            params = {name: param for name, param in model.named_parameters()}
            buffers = {name: buffer for name, buffer in model.named_buffers()}

            def compute_loss_functional(params, buffers, sample_input, sample_label):
                sample_input = sample_input.unsqueeze(0)
                sample_label = sample_label.unsqueeze(0)
                y_hat = functional_call(model, (params, buffers), (sample_input,))
                return compute_loss(y_hat, sample_label)

        per_sample_grads_dict = vmap(grad(compute_loss_functional), in_dims=(None, None, 0, 0))(params, buffers, batch, labels)

        with torch.no_grad():
            for (name, param), mapping in zip(model.named_parameters(), rand_mapping):
                param_grads = per_sample_grads_dict[name].view(batch.shape[0], -1)  # [batch_size, param_size]
                db[i:i_max] += param_grads @ mapping  # [batch_size, subspace_size]

    g_overall = [torch.zeros_like(p) for p in base_model.parameters()]
    for b in range(batches):
        idx = dataset[b]
        batch = input_ids[idx]
        labels = total_labels[idx]

        y_hat = model(batch)
        loss = compute_loss(y_hat, labels)
        g = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)

        for g_delta, param, g_cum in zip(g, model.parameters(), g_overall):
            param.data -= LR * g_delta.data
            g_cum.data += LR * g_delta.data

        if b % 5 == 0:
            update_db(model)


    for b in range(batches):
        subset_db = db[dataset[b]]

        g = torch.sum(subset_db, dim=0)

        scores = (g - subset_db) @ query
        removed = scores.argmin().item()
        removed_id = dataset[b][removed].item()


        # breakpoint()
        scores = (db + g - subset_db[removed]) @ query
        used_indices = torch.tensor(list(used), dtype=torch.long, device=device)
        scores[used_indices] = 1000
        selected = scores.argmin().item()
        dataset[b][removed] = selected
        # breakpoint()
        used.add(selected)
        used.remove(removed_id)


    distance = sum([((pg + pd)**2).sum() for pg, pd in zip(g_overall, diffs)])
    diff_norm = sum([((pd)**2).sum() for pg, pd in zip(g, diffs)])
    non_ignorable = [(torch.abs(pd) > 0.005) for pd in diffs]
    non_ig = sum([p.float().sum() for p in non_ignorable])
    positives = sum(((pg * pd * ig.float()) > 0).sum() for pg, pd, ig in zip(g_overall, diffs, non_ignorable))
    print('distance', distance.item())
    print('diff', diff_norm.item())
    print('positives', positives.item() / non_ig.item())
    print('dataset', len(used))
