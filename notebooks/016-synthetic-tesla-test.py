import copy

import torch
import torch.nn.functional as F
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
finetuned_ckp = '/network/scratch/b/brownet/hydra-runs/finetune-grammar/checkpoints/finetuning_step_00070'
load_weights_from_checkpoint(finetuned_model, finetuned_ckp, model_name='transformer')


batches = 1000
batch_size = 32

# batch_size = batches * batch_size // 4
# batches = 1
seq_len = 200

vocab_size = base_model.tok_emb.num_embeddings

input_ids = torch.zeros((batches * batch_size, seq_len), dtype=torch.long, device=device)
input_ids[:, 0] = torch.randint(1, vocab_size-1, (batches * batch_size,), device=device)



input_ids.shape
for i in range(1, seq_len):
    with torch.no_grad():
        outputs = finetuned_model(input_ids[:, :i])[:, -1]
        dist = torch.distributions.Categorical(logits=outputs)
        samples = dist.sample()
        input_ids[:, i] = samples

data = F.one_hot(input_ids, num_classes=vocab_size).to(dtype=torch.float32)
x_star = data.clone().requires_grad_(True).view(batches * batch_size, seq_len, vocab_size)

y_star = torch.zeros_like(x_star)
for i in range(0, batches * batch_size, 100):
    i_max = min(batches * batch_size, i + 100)
    x = x_star[i:i_max]
    y_star[i:i_max] = finetuned_model.forward_with_continuous_inputs(x).detach().clone()

x_star = x_star.view(batches, batch_size, seq_len, vocab_size).requires_grad_(True)
y_star = y_star.view(batches, batch_size, seq_len, vocab_size).requires_grad_(True)


for name, param in base_model.named_parameters():
    param.requires_grad_(True)
    param.grad = None

META_BATCHES = 1
LR = 0.00001
META_LR = 0.00000

diffs = [p_fine - p_pre for p_fine, p_pre in zip(finetuned_model.parameters(), base_model.parameters())]
diffs_unmodified = [p_fine - p_pre for p_fine, p_pre in zip(finetuned_model.parameters(), base_model.parameters())]

def compute_loss(y_hat, y_star):
    temperature = 1.0
    log_probs_student = F.log_softmax(y_hat / temperature, dim=-1)
    probs_teacher = F.softmax(y_star / temperature, dim=-1)
    kl_loss = F.kl_div(log_probs_student, probs_teacher, reduction="batchmean") * (temperature ** 2)
    return kl_loss

g_overall = [torch.zeros_like(p) for p in base_model.parameters()]
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    ckp = base_model
    for b in tqdm(range(batches)):
        for _ in range(META_BATCHES):
            x_grad = torch.zeros_like(x_star)
            y_grad = torch.zeros_like(y_star)
            model = copy.deepcopy(ckp)
            full_loss = 0.0

            # for i in range(2):
            g = [torch.zeros_like(p) for p in model.parameters()]
            y_hat = model.forward_with_continuous_inputs(x_star[b])
            assert y_hat.shape == y_star[b].shape
            loss = compute_loss(y_hat, y_star[b])

            meta_loss = 0.0
            # breakpoint()
            g_actuals = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
            for g_delta, param, diff, g_cum in zip(g_actuals, model.parameters(), diffs, g):
                meta_loss += ((diff + LR* g_delta)**2).sum() # + 2 * (LR ** 2) * (g_cum * g_delta).sum()
                # param.data -= LR * dg_elta
                g_cum += LR * g_delta
                # meta_loss += torch.cosine_similarity((diff + g_cum.data).view(1, -1), g_delta.view(1, -1)) + 1
                # breakpoint()


            # new_params = {}
            # for g_delta, (name, param), diff, g_cum in zip(g_actuals, model.named_parameters(), diffs, g):
            #     # breakpoint()
            #     param -= LR * g_delta
                # breakpoint()
                # pre = model.state_dict()[name]
                # breakpoint()
                # assert pre is param

                # breakpoint()
                # setattr(model, name, param - LR * g_delta)
                # new_params[name] = param - LR * g_delta

                # breakpoint()
                # assert (pre != model.state_dict()[name]).any()
                # g_cum += LR * g_delta.data
            # breakpoint()
            # new_params['lm_head.weight'] = new_params['tok_emb.weight']
            # breakpoint()
            # model.load_state_dict(new_params)
            # breakpoint()

            # for ft_params, param, diff, g_cum in zip(finetuned_model.parameters(), model.parameters(), diffs, diffs):
            #     full_loss += ((ft_params - param)**2).sum() # + 2 * (LR ** 2) * (g_cum * g_delta).sum()
            full_loss = meta_loss

            # breakpoint()
            x_grad, y_grad = torch.autograd.grad(full_loss, [x_star, y_star], retain_graph=False, create_graph=False)

            with torch.no_grad():
                x_star.data -= META_LR * x_grad.data
                y_star.data -= META_LR * y_star.data
            print('loss', full_loss.item())

            distance = sum([((pg + pd)**2).sum() for pg, pd in zip(g, diffs)])
            diff_norm = sum([((pd)**2).sum() for pg, pd in zip(g, diffs)])
            non_ignorable = [(torch.abs(pd) > 0.005) for pd in diffs_unmodified]
            non_ig = sum([p.float().sum() for p in non_ignorable])
            # print('non-ig', non_ig)
            # breakpoint()
            # breakpoint()
            positives = sum(((pg * pd * ig.float()) > 0).sum() for pg, pd, ig in zip(g_overall, diffs_unmodified, non_ignorable))
            print('distance', distance.item())
            print('diff', diff_norm.item())
            print('positives', positives.item() / non_ig.item())
            # if (_  + 1) % 20 == 0:
            # breakpoint()

        # breakpoint()
        y_hat = ckp.forward_with_continuous_inputs(x_star[b])
        loss = compute_loss(y_hat, y_star[b])
        g_actuals = torch.autograd.grad(loss, ckp.parameters(), retain_graph=False, create_graph=False)
        for g_delta, param, diff, g_cum in zip(g_actuals, ckp.parameters(), diffs, g_overall):
            param.data -= LR * g_delta.data
            diff.data += LR * g_delta.data
            g_cum.data += LR * g_delta.data
        # if (b + 1) % 10 == 0:
        #     breakpoint()

    # breakpoint()
