import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import to_numpy


@torch.jit.script
def round_ste_func(x):
    return (torch.round(x) - x).detach() + x


@torch.jit.script
def l0_sample(
    log_alpha,
    beta: float = 2.0 / 3.0,
    zeta: float = 1.1,
    gamma: float = -0.1,
    N: int = 1,
    device: torch.device | None = None,
):
    size = N if N > 1 else log_alpha.size(0)
    u = torch.rand(size, device=device)
    sigm = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    sbar = sigm * (zeta - gamma) + gamma
    return torch.clamp(sbar, 0.0, 1.0)


@torch.jit.script
def hardsigmoid(
    x,
    beta: float = 2.0 / 3.0,
    zeta: float = 1.1,
    gamma: float = -0.1,
    N: int | None = None,
    device: torch.device | None = None,
):
    sigm = torch.sigmoid(x / beta)
    ybar = sigm * (zeta - gamma) + gamma
    return torch.clamp(ybar, 0.0, 1.0)


@torch.jit.script
def _pact_clip_jit(x, x_max, signed: int):
    x_max_scaled = (1.0 - 1e-7) * x_max
    x_min = -x_max_scaled if signed != 0 else torch.zeros_like(x_max)
    return -F.relu(x_max_scaled - x_min - F.relu(x - x_min)) + x_max_scaled


def invsigmoid(x, beta=2.0 / 3.0):
    return -beta * np.log(1.0 / x - 1)


@torch.jit.script
def hc_prob_pos(p, beta: float = 2.0 / 3.0, zeta: float = 1.1, gamma: float = -0.1):
    return torch.sigmoid(p - beta * math.log(-gamma / zeta))


def relu(x):
    return x * (x >= 0.0)


@torch.jit.script
def _eager_recursive_quantize_ladder(
    x,
    s2,
    s4,
    s8,
    s16,
    s32,
    g4,
    g8,
    g16,
    g32,
    training: bool,
):
    # s2 stage
    inv_s2 = 1.0 / s2
    x_div_s2 = x * inv_s2

    if training:
        x_q = s2 * round_ste_func(x_div_s2)
    else:
        x_q = s2 * torch.round(x_div_s2)

    # Normalize gates to tensors on the right device/dtype
    # JIT requires explicit types or tensors
    # We assume g4 etc are tensors or floats.
    # But JIT doesn't like mixed types.
    # Let's assume they are passed as tensors from recursive_quantization

    # s4 stage
    residual = x - x_q
    r_div_s4 = residual * (1.0 / s4)
    if training:
        r_round = round_ste_func(r_div_s4)
    else:
        r_round = torch.round(r_div_s4)
    x_q = x_q + g4 * s4 * r_round

    # s8 stage
    residual = x - x_q
    r_div_s8 = residual * (1.0 / s8)
    if training:
        r_round = round_ste_func(r_div_s8)
    else:
        r_round = torch.round(r_div_s8)
    x_q = x_q + g4 * g8 * s8 * r_round

    # s16 stage
    residual = x - x_q
    r_div_s16 = residual * (1.0 / s16)
    if training:
        r_round = round_ste_func(r_div_s16)
    else:
        r_round = torch.round(r_div_s16)
    x_q = x_q + g4 * g8 * g16 * s16 * r_round

    # s32 stage
    residual = x - x_q
    r_div_s32 = residual * (1.0 / s32)
    if training:
        r_round = round_ste_func(r_div_s32)
    else:
        r_round = torch.round(r_div_s32)
    x_q = x_q + g4 * g8 * g16 * g32 * s32 * r_round

    return x_q


def get_x_min_x_max(x, use_mse=False):
    x_min, x_max = x.min(), x.max()
    if not use_mse or True:
        return x_min, x_max

    # Ugly hack: if we use MSE then it's 4 bits.
    signed = x_min < 0

    x_absmax = max(abs(x_min), x_max)

    x = x.data.cpu().numpy().flatten()

    min_mse = np.inf
    best_xmax = None
    maxes = np.linspace(0, x_absmax, 1000)[1:]
    for x_max in maxes:
        xmx = (1 - 1e-7) * x_max
        x_min = -xmx if signed else 0.0
        x_clipped = -relu(xmx - x_min - relu(x - x_min)) + xmx

        x_min = -xmx if signed else 0.0
        s = (x_max - x_min) / (2**4 - 1)

        xq = s * np.round(x_clipped / s)

        mse = np.mean((x - xq) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_xmax = x_max

    assert best_xmax

    best_xmin = -best_xmax if signed else 0
    return best_xmin, best_xmax


def rank_prior(probs):
    # Vectorized cumulative product to reduce Python-loop overhead in gate sampling
    if probs.dim() == 1:
        return torch.cumprod(probs, dim=0)
    # Fallback for unexpected shapes
    flat = probs.view(-1)
    return torch.cumprod(flat, dim=0)


def recursive_quantization(
    x,
    s_2,
    s_4,
    s_8,
    s_16,
    s_32,
    gate_2,
    gate_4,
    gate_8,
    gate_16,
    gate_32,
    round_f,
    quantize: bool,
    training=False,
):

    round_f = round_f or round_ste_func

    if quantize:
        gate_4, gate_8, gate_16, gate_32 = (
            torch.as_tensor(g if g is not None else 0.0, device=x.device, dtype=x.dtype)
            for g in [gate_4, gate_8, gate_16, gate_32]
        )

        x_q = _eager_recursive_quantize_ladder(
            x, s_2, s_4, s_8, s_16, s_32, gate_4, gate_8, gate_16, gate_32, training
        )
    else:
        x_q = x

    if gate_2 is not None:
        dummy = torch.ones((1, 1), device=x.device)
        x_q = torch.cat([dummy, round_f(gate_2)]) * x_q

    del x

    return x_q


class BayesianBitsQuantizer(nn.Module):
    def __init__(
        self,
        n_bits,
        x_min=None,
        x_max=None,
        device="cuda",
        gating_method="l0",
        gamma_2_init=invsigmoid(0.9),
        gamma_4_init=invsigmoid(0.9),
        gamma_8_init=invsigmoid(0.9),
        gamma_16_init=invsigmoid(0.9),
        gamma_32_init=invsigmoid(0.9),
        learned_scale=None,
        clip_input=False,
        act_quant=False,
        checkpointing=False,
        include_pruning=False,
        prune_only=False,
        fixed_8bit=False,
        fixed_4bit=False,
        fixed_full_precision=False,
        reg_type="const",
        prune_rank=False,
    ):
        # learned_scale can be None, 'scale' or 'range'
        super(BayesianBitsQuantizer, self).__init__()
        assert n_bits == 16

        self.device = device

        self.n_bits = n_bits
        self.learned_scale = learned_scale
        self.clip_input = clip_input
        self.act_quant = act_quant
        self.checkpointing = checkpointing
        self.include_pruning = include_pruning
        self.prune_only = prune_only
        assert (
            not self.prune_only or self.include_pruning
        ), f"{self.prune_only}; {self.include_pruning}; {self.act_quant}"

        self.fixed_8bit = fixed_8bit
        self.fixed_4bit = fixed_4bit
        self.fixed_full_precision = fixed_full_precision
        self.mac_count, self.max_macs = None, None
        self.do_reg = True
        self.reg_type = reg_type

        self.init_ranges = False
        self.mini = self.maxi = None
        self.x_min = self.x_max = self.x_r = None
        self.signed = None
        self.s_2 = self.s_4 = self.s_8 = self.s_16 = self.s_32 = self.signed = None
        self.init_xminmax(
            x_min or -10, x_max or 10, final_init=(None not in [x_min, x_max])
        )

        self.gamma_2_init = 6
        self.gamma_16_init = gamma_16_init
        self.gamma_32_init = gamma_32_init

        gamma_4 = torch.Tensor([gamma_4_init]).to(device)
        gamma_8 = torch.Tensor([gamma_8_init]).to(device)
        gamma_16 = torch.Tensor([gamma_16_init]).to(device)
        gamma_32 = torch.Tensor([gamma_32_init]).to(device)

        self.gamma_4 = torch.nn.Parameter(gamma_4)
        self.gamma_8 = torch.nn.Parameter(gamma_8)
        self.gamma_16 = torch.nn.Parameter(gamma_16)
        self.gamma_32 = torch.nn.Parameter(gamma_32)
        self.register_parameter("gamma_2", None)

        self.gating_method = gating_method
        self.hc_beta = 2.0 / 3.0
        self.hc_gamma, self.hc_zeta = -0.1, 1.1
        self.hc_thres = 0.34
        self.prune_rank = prune_rank
        self.curr_rank = None

    @property
    def gmp_params(self):
        res = self.gamma_4, self.gamma_8, self.gamma_16, self.gamma_32
        if self.learned_scale == "scale":
            res += (self.s_2,)
        elif self.learned_scale == "range":
            res += self.x_max, self.x_min

        return res

    def init_xminmax(self, x_min, x_max, final_init=True):
        if x_min is None or self.init_ranges:
            return

        # Ranges not yet initialized
        min_value = min(float(x_min), 0)

        if self.x_min is None:
            x_min = torch.Tensor([min_value]).to(self.device)
            x_max = torch.Tensor([max(float(x_max), 0)]).to(self.device)

            # Make parameters if we learn the range, otherwise these are just regular old Tensors
            self.x_min = nn.Parameter(x_min) if self.learned_scale == "range" else x_min
            self.x_max = nn.Parameter(x_max) if self.learned_scale == "range" else x_max

            if self.learned_scale == "scale":
                s_2 = self.compute_base_scale()
                self.s_2 = nn.Parameter(s_2)

        else:
            # print('Updating existing parameters (instead of creating new ones)')
            assert self.x_max is not None
            self.x_min.data.fill_(min_value)
            self.x_max.data.fill_(max(float(x_max), 0))

            if self.learned_scale == "scale":
                s_2 = self.compute_base_scale()
                self.s_2.data.fill_(float(s_2))

        self.init_ranges = final_init
        if x_min == 0:
            self.x_min.requires_grad = False

        self.signed = int(self.x_min.data[0] < 0)
        self.mini, self.maxi = (
            (-(2 ** (2 - 1)), 2 ** (2 - 1) - 1) if self.signed else (0, 2**2 - 1)
        )

    def compute_base_scale(self):
        if self.x_max > 0:
            x_min = -self.x_max if self.signed else 0.0
            s_2 = (self.x_max - x_min) / (2**2 - 1)
        elif self.x_min < 0:
            x_max = -self.x_min
            s_2 = (x_max - self.x_min) / (2**2 - 1)
        else:
            print(self.x_min, self.x_max)
        return s_2

    def get_base_scale(self):
        if self.learned_scale == "scale":
            return self.s_2
        else:
            return self.compute_base_scale()

    def get_gates(self, N=1):
        beta, zeta, gamma = self.hc_beta, self.hc_zeta, self.hc_gamma
        args, kwargs = [], {}
        offset = math.log(-self.hc_gamma / self.hc_zeta) * self.hc_beta
        if self.training:
            kwargs = dict(N=N)
            if self.gating_method == "l0":
                gate_f = l0_sample
                args = beta, zeta, gamma

            elif self.gating_method == "fixed":
                gate_f = hardsigmoid
                args = beta, zeta, gamma

            else:
                raise ValueError(
                    "Bayesian Bits quantizer only works with "
                    f"[l0, fixed] gates, not {self.gating_method}"
                )

        else:

            def gate_f(x, **kwargs):
                return (torch.sigmoid(offset - x) < self.hc_thres).float()

        if self.prune_rank:
            probs = gate_f(self.gamma_2, device=self.device, *args, **kwargs)
            gate_2 = rank_prior(probs)
            self.curr_rank = torch.count_nonzero(torch.round(gate_2) > 0) + 1
        else:
            gate_2 = (
                gate_f(self.gamma_2, device=self.device, *args, **kwargs)
                if self.include_pruning
                else None
            )

        gammas = torch.stack([self.gamma_4, self.gamma_8, self.gamma_16, self.gamma_32])
        active_mask = gammas > -2
        if active_mask.any():
            gate_vals = torch.zeros_like(gammas)
            gate_vals[active_mask] = gate_f(
                gammas[active_mask], device=self.device, *args, **kwargs
            )
            gate_4, gate_8, gate_16, gate_32 = gate_vals
        else:
            zero = torch.zeros_like(self.gamma_4)
            gate_4 = gate_8 = gate_16 = gate_32 = zero

        return gate_2, gate_4, gate_8, gate_16, gate_32

    def pact_clip(self, x):
        return _pact_clip_jit(x, self.x_max, self.signed)

    def scale_clip(self, x):
        # in this case, self.s_2 is the learnable parameter
        self.x_r = ((2**2 - 1) * self.s_2).to(self.device)
        self.x_min = self.signed * (-self.x_r)
        self.x_max = (1 - 1e-7) * self.x_r  # ensure nothing gets rounded to 1.5
        return self.pact_clip(x)

    def get_scales(self):
        s_2 = self.get_base_scale()
        s_4 = s_2 / (2**2 + 1)
        s_8 = s_4 / (2**4 + 1)
        s_16 = s_8 / (2**8 + 1)
        s_32 = s_16 / (2**16 + 1)
        return s_2, s_4, s_8, s_16, s_32

    def get_bops(self, bw):
        if self.mac_count is None or self.reg_type == "const":
            # print('found non-initialized mac count!')
            # self.mac_count = 1.
            return 1.0
        return (self.mac_count / self.max_macs) * bw

    def regularizer(self):
        if not self.do_reg:
            zeros = torch.Tensor([0.0])
            zeros = zeros.to(self.device)
            return zeros
        ones = torch.Tensor([1.0])
        ones = ones.to(self.device)

        beta, zeta, gamma = self.hc_beta, self.hc_zeta, self.hc_gamma
        q_b2 = (
            hc_prob_pos(self.gamma_2, beta, zeta, gamma)
            if self.include_pruning or self.prune_rank
            else ones
        )
        q_b4 = hc_prob_pos(self.gamma_4, beta, zeta, gamma)
        q_b8 = hc_prob_pos(self.gamma_8, beta, zeta, gamma)
        q_b16 = hc_prob_pos(self.gamma_16, beta, zeta, gamma)
        q_b32 = hc_prob_pos(self.gamma_32, beta, zeta, gamma)

        kl_b2 = (
            self.get_bops(2) * q_b2 if self.include_pruning or self.prune_rank else ones
        )
        kl_b4 = q_b2 * self.get_bops(4) * q_b4
        kl_b8 = q_b2 * q_b4 * self.get_bops(8) * q_b8
        kl_b16 = q_b2 * q_b4 * q_b8 * self.get_bops(16) * q_b16
        kl_b32 = q_b2 * q_b4 * q_b8 * q_b16 * self.get_bops(32) * q_b32

        reg_quant = kl_b4.sum() + kl_b8.sum() + kl_b16.sum() + kl_b32.sum()
        reg_quant = reg_quant / (
            self.gamma_2.size(0) if self.include_pruning or self.prune_rank else 1.0
        )

        if self.include_pruning or self.prune_rank:
            reg_rank = kl_b2.sum()
            reg_rank = reg_rank / self.gamma_2.size(0)
        else:
            reg_rank = torch.zeros_like(reg_quant)
        return reg_quant, reg_rank

    def _forward_impl(self, x):
        if self.x_min is None or not self.init_ranges:
            x_min, x_max = get_x_min_x_max(x, use_mse=self.fixed_4bit)
            self.init_xminmax(x_min, x_max)

        if (
            (self.prune_rank or self.include_pruning)
            and self.gamma_2 is None
            and not self.act_quant
        ):
            print(
                f"Initializing gates of size {x.size(0)} for pruning of tensor with size {x.size()}"
            )

            gamma_2 = torch.Tensor((x.size(0) - 1) * [self.gamma_2_init])
            self.gamma_2 = torch.nn.Parameter(gamma_2.to(self.device))

        if self.prune_only and not (self.fixed_8bit or self.fixed_4bit):
            pass
        else:
            if self.clip_input == "range":
                x = self.pact_clip(x)
            elif self.clip_input == "scale":
                x = self.scale_clip(x)

        s_2, s_4, s_8, s_16, s_32 = self.get_scales()

        gates = self.get_gates()
        gate_2 = gates[0]
        gate_4, gate_8, gate_16, gate_32 = (None if g == 0 else g for g in gates[1:])

        if self.fixed_8bit:
            x = s_8 * round_ste_func(x / s_8)
        elif self.fixed_4bit:
            x = s_4 * round_ste_func(x / s_4)

        if self.prune_only:
            assert gate_2 is not None
            rest_shape = [1] * (len(x.size()) - 1)
            rshp = lambda _x: _x.view(_x.size(0), *rest_shape)
            gate_2 = rshp(gate_2)
            x = gate_2 * x
            return x

        if gate_2 is not None:
            rest_shape = [1] * (len(x.size()) - 1)
            rshp = lambda _x: _x.view(_x.size(0), *rest_shape)
            gate_2 = rshp(gate_2)

        round_f = (
            None
            if self.checkpointing
            else round_ste_func if self.training else torch.round
        )

        if self.training and self.checkpointing:
            return checkpoint(
                recursive_quantization,
                x,
                s_2,
                s_4,
                s_8,
                s_16,
                s_32,
                gate_2,
                gate_4,
                gate_8,
                gate_16,
                gate_32,
                round_f,
                not (self.fixed_4bit or self.fixed_8bit or self.fixed_full_precision),
                self.training,
                preserve_rng_state=False,
            )
        else:
            return recursive_quantization(
                x,
                s_2,
                s_4,
                s_8,
                s_16,
                s_32,
                gate_2,
                gate_4,
                gate_8,
                gate_16,
                gate_32,
                round_f,
                not (self.fixed_4bit or self.fixed_8bit or self.fixed_full_precision),
                self.training,
            )

    def forward(self, x):
        return self._forward_impl(x)


class Quantizer(nn.Module):
    """A general quantizer class which gets a quantization function and then takes care of
    the statistics etc. Depending on the parameters it keeps a running mean or takes the current
    matrix to get the necessary statistics to initialize the quantization function.

    Parameters
    ----------
    method: str
        Name of the quantization method to use.
    n_bits:
        Number of bits for quantization.
    use_running_mean:
        If true it keeps a running mean of the matrix statistics, otherwise it uses alwats the
        statistics of the current matrix or quantization.
    momentum:
        The momentum for the running mean.

    """

    def __init__(
        self,
        method,
        n_bits,
        use_running_mean=True,
        momentum=0.1,
        percentile=None,
        fix_range_on_first_forward=True,
        scale_domain="linear",
        gating_method="l0",
        gamma_2_init=2.0,
        gamma_4_init=2.0,
        gamma_8_init=2.0,
        gamma_16_init=2.0,
        gamma_32_init=2.0,
        clip_input=False,
        learned_scale=None,
        act_quant=False,
        checkpointing=False,
        include_pruning=False,
        prune_only=False,
        fixed_8bit=False,
        fixed_4bit=False,
        fixed_full_precision=False,
        reg_type="const",
        prune_rank=False,
        device="cuda",
    ):
        super(Quantizer, self).__init__()

        self.method = method
        self.scale_domain = scale_domain
        self.n_bits = n_bits
        self.use_running_mean = use_running_mean
        self.momentum = momentum
        self.percentile = percentile
        self.act_quant = act_quant
        self.checkpointing = checkpointing
        self.include_pruning = include_pruning
        self.prune_only = prune_only
        assert (
            not prune_only or include_pruning
        ), f"{prune_only}; {include_pruning}; {act_quant}"
        self.prune_rank = prune_rank
        self.fixed_8bit = fixed_8bit
        self.fixed_4bit = fixed_4bit
        self.fixed_full_precision = fixed_full_precision
        self.reg_type = reg_type

        self.x_min = None
        self.x_max = None
        self.quantizer = None
        self.fixed_range = False
        self.fix_range_on_first_forward = fix_range_on_first_forward

        self.gating_method = gating_method
        self.gamma_2_init = gamma_2_init
        self.gamma_4_init = gamma_4_init
        self.gamma_8_init = gamma_8_init
        self.gamma_16_init = gamma_16_init
        self.gamma_32_init = gamma_32_init

        self._on = True
        self.owner = None
        self.name = None

        self.clip_input = clip_input
        self.learned_scale = learned_scale

        self.device = device

    def off(self):
        self._on = False
        self.quantizer.do_reg = False

    def on(self):
        self._on = True
        self.quantizer.do_reg = True

    def create_quantizer(self):
        if self.method == "bayesian_bits":
            assert (
                not self.prune_only or self.include_pruning
            ), f"{self.prune_only}; {self.include_pruning}; {self.act_quant}"
            q = BayesianBitsQuantizer(
                self.n_bits,
                self.x_min,
                self.x_max,
                device=self.device,
                gating_method=self.gating_method,
                gamma_2_init=self.gamma_2_init,
                gamma_4_init=self.gamma_4_init,
                gamma_8_init=self.gamma_8_init,
                gamma_16_init=self.gamma_16_init,
                gamma_32_init=self.gamma_32_init,
                clip_input=self.clip_input,
                learned_scale=self.learned_scale,
                act_quant=self.act_quant,
                checkpointing=self.checkpointing,
                include_pruning=self.include_pruning,
                prune_only=self.prune_only,
                fixed_8bit=self.fixed_8bit,
                fixed_4bit=self.fixed_4bit,
                fixed_full_precision=self.fixed_full_precision,
                reg_type=self.reg_type,
                prune_rank=self.prune_rank,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return q

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def fix_quant_ranges(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
        self.fixed_range = True
        self.quantizer = self.create_quantizer()

    def forward(self, x_float):
        if not self._on:
            return x_float

        if "bayesian_bits" in self.method or "pact_only" in self.method:
            if self.quantizer is None:
                self.quantizer = self.create_quantizer()
        else:
            print(
                f"DEBUG: Quantizer {self.name} method={self.method} calling update_quantization_range"
            )
            self.update_quantization_range(x_float)
        return self.quantizer(x_float)

    def update_quantization_range(self, data):
        if (
            not self.training and self.use_running_mean and self.quantizer
        ) or self.fixed_range:
            # assert self.quantizer is not None
            # We are in evaluation and quantization ranges are not updated => no op
            assert self.quantizer is not None
        else:
            x_min, x_max = self.get_x_min_x_max(data)
            if self.use_running_mean:
                if self.training:
                    self.x_min = self.momentum * x_min + (1 - self.momentum) * (
                        self.x_min or x_min
                    )
                    self.x_max = self.momentum * x_max + (1 - self.momentum) * (
                        self.x_max or x_max
                    )
                elif self.x_min is None:
                    print(
                        "Warning: running mean not initialized, using first batch for quantization"
                    )
                    self.x_min = x_min
                    self.x_max = x_max
                    # raise ValueError("Running mean not initialized.")
            else:
                self.x_min = x_min
                self.x_max = x_max

            if self.fix_range_on_first_forward:
                self.fix_quant_ranges(self.x_min, self.x_max)
            else:
                self.quantizer = self.create_quantizer()

    def get_x_min_x_max(self, data):
        if self.percentile:
            data_np = to_numpy(data)
            x_min, x_max = np.percentile(
                data_np, (self.percentile, 100 - self.percentile)
            )
        else:
            x_min = min(0.0, float(data.min()))
            x_max = max(0.0, float(data.max()))

        return x_min, x_max

    def extra_repr(self):
        s = "{method}, n_bits={n_bits}, use_running_mean={use_running_mean}"
        if self.use_running_mean:
            s += ", momentum={momentum}"
        if self.percentile:
            s += ", percentile={percentile}"
        return s.format(**self.__dict__)


class PerChannelQuantizeWrapper(nn.Module):
    """A wrapper that does a per channel quantization (one set of parameters per channel).

    Parameters
    ----------
    quantizer: Quantizer
        An initialized quantizer that will be used per channel. This will be cloned for each channel
        so it keeps track of their own statistics.

    """

    def __init__(self, quantizer):
        super().__init__()
        self.org_quantizer = quantizer
        self.channel_quantizers = None

    def forward(self, x_float):
        if self.channel_quantizers is None:
            self.channel_quantizers = [
                copy.deepcopy(self.org_quantizer) for _ in range(x_float.shape[0])
            ]

        x_quant = torch.zeros_like(x_float)
        for idx in range(x_float.shape[0]):
            x_quant[idx] = self.channel_quantizers[idx](x_float[idx])
        return x_quant


class QuantizationHijacker(nn.Module):
    """Mixin class that 'hijacks' the forward pass in a module to perform quantization and
    dequantization on the weights and output distributions.

    Usage:
    To make a quantized nn.Linear layer:
    class HijackedLinear(QSchemeForwardHijacker, nn.Linear):
        pass

    It is vital that QSchemeForwardHijacker is the first parent class, and that the second parent
    class derives from nn.Module, otherwise it will not be reached by a super(., .) call.

    NB: this implementation (for now) assumes that there will always be some training involved,
    e.g. to estimate the activation ranges.

    """

    def __init__(
        self,
        *args,
        method="qscheme",
        n_bits=8,
        n_bits_act=None,
        per_channel_weights=False,
        act_momentum=0.1,
        percentile=None,
        activation=None,
        fix_range_on_first_forward=False,
        scale_domain="linear",
        gating_method="l0",
        gamma_2_init=2.0,
        gamma_4_init=2.0,
        gamma_8_init=2.0,
        gamma_16_init=2.0,
        gamma_32_init=2.0,
        learned_scale=False,
        clip_input=False,
        checkpointing=False,
        include_pruning=False,
        reg_type="const",
        prune_only=False,
        fixed_8bit=False,
        fixed_48bit=False,
        fixed_full_precision=False,
        device="cuda",
        **kwargs,
    ):
        super(QuantizationHijacker, self).__init__(*args, **kwargs)
        self.act_momentum = act_momentum
        self.activation = activation

        self.device = device

        if self.activation is None or self.activation.upper() == "NONE":
            self.activation_function = None
        elif self.activation.upper() == "RELU":
            self.activation_function = nn.ReLU()
        elif self.activation.upper() == "HARDTANH":
            self.activation_function = nn.Hardtanh()
        elif self.activation.upper() == "RELU6":
            self.activation_function = nn.ReLU6()
        elif self.activation.upper() == "SIGMOID":
            self.activation_function = nn.Sigmoid()
        else:
            raise ValueError(
                "With quantization: please only use ReLU, HardTanH or Sigmoid"
            )

        self.cached_params = None

        self.quant_params = None
        self.N = n_bits
        self.n_bits_act = n_bits_act or n_bits
        self.act_moving_tgt = None
        self._quant_w = False
        self._quant_a = False

        # self.activation_quantizer = Quantizer(
        #     method,
        #     self.n_bits_act,
        #     True,
        #     act_momentum,
        #     device=device,
        #     fix_range_on_first_forward=fix_range_on_first_forward,
        #     scale_domain=scale_domain,
        #     gating_method=gating_method,
        #     gamma_2_init=gamma_2_init,
        #     gamma_4_init=gamma_4_init,
        #     gamma_8_init=gamma_8_init,
        #     gamma_16_init=gamma_16_init,
        #     gamma_32_init=gamma_32_init,
        #     learned_scale=learned_scale,
        #     clip_input=clip_input,
        #     fixed_8bit=fixed_8bit,
        #     fixed_4bit=fixed_48bit,
        #     fixed_full_precision=fixed_full_precision,
        #     act_quant=True,
        #     checkpointing=checkpointing,
        #     reg_type=reg_type,
        # )
        self.activation_quantizer = None
        assert not prune_only or include_pruning, f"{prune_only}; {include_pruning}"
        self.weight_quantizer = Quantizer(
            method,
            self.N,
            False,
            device=device,
            percentile=percentile,
            fix_range_on_first_forward=fix_range_on_first_forward,
            scale_domain=scale_domain,
            gating_method=gating_method,
            gamma_2_init=gamma_2_init,
            gamma_4_init=gamma_4_init,
            gamma_8_init=gamma_8_init,
            gamma_16_init=gamma_16_init,
            gamma_32_init=gamma_32_init,
            learned_scale=learned_scale,
            clip_input=clip_input,
            checkpointing=checkpointing,
            include_pruning=include_pruning,
            reg_type=reg_type,
            prune_only=prune_only,
            fixed_8bit=fixed_8bit,
            fixed_4bit=fixed_48bit,
            fixed_full_precision=fixed_full_precision,
        )

        self.activation_save_target = None
        self.activation_save_name = None

        if per_channel_weights:
            self.weight_quantizer = PerChannelQuantizeWrapper(self.weight_quantizer)

    def quantized_weights(self):
        self.cached_params = None
        self._quant_w = True

    def full_precision_weights(self):
        self.cached_params = None
        self._quant_w = False

    def quantized_acts(self):
        self._quant_a = True
        self.activation_quantizer.do_reg = True

    def full_precision_acts(self):
        self._quant_a = False
        self.activation_quantizer.do_reg = False

    def quantized(self):
        self.quantized_weights()
        self.quantized_acts()

    def full_precision(self):
        self.full_precision_weights()
        self.full_precision_acts()

    @property
    def is_conv(self):
        return isinstance(self, (nn.Conv1d, nn.Conv2d))

    def forward(self, x, offsets=None):
        weight, bias = self.get_params()
        res = self.run_forward(x, weight, bias, offsets=offsets)
        res = self.quantize_activations(res)
        return res

    def train(self, mode=True):
        super(QuantizationHijacker, self).train(mode)
        if mode:
            self.cached_params = None
        return self

    def get_params(self):
        if not self.training and self.cached_params:
            return self.cached_params

        weight, bias = self.get_weight_bias()

        if self._quant_w:
            weight = self.weight_quantizer(weight)
            if bias is not None:
                bias = self.weight_quantizer(bias)

        if not self.training and self.cached_params is None:
            self.cached_params = (
                torch.Tensor(to_numpy(weight)).to(weight.device),
                (
                    torch.Tensor(to_numpy(bias)).to(bias.device)
                    if bias is not None
                    else None
                ),
            )

        return weight, bias

    def get_weight_bias(self):
        bias = None
        if hasattr(self, "bias"):
            bias = self.bias
        return self.weight, bias

    def run_forward(self, x, weight, bias, detach=False, offsets=None):
        if detach:
            args = x.detach().contiguous(), weight.detach().contiguous()
        else:
            args = x.contiguous(), weight.contiguous()
        bias = bias.detach() if bias is not None and detach else bias

        if isinstance(self, nn.Linear):
            return F.linear(*args, bias=bias)
        elif isinstance(self, (nn.Conv1d, nn.Conv2d)):
            kwargs = dict(
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            if isinstance(self, nn.Conv1d):
                return F.conv1d(*args, **kwargs)
            if isinstance(self, nn.Conv2d):
                return F.conv2d(*args, **kwargs)
            # TODO: add support for Conv3d, should be trivial
        elif isinstance(self, nn.EmbeddingBag):
            cont_offseargsts = None
            if detach:
                args = weight.detach().contiguous(), x.detach().contiguous
                if offsets is not None:
                    cont_offsets = offsets.detach().contiguous()
            else:
                args = weight.contiguous(), x.contiguous()
                if offsets is not None:
                    cont_offsets = offsets.contiguous()
            return F.embedding_bag(
                *args,
                cont_offsets,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.mode,
                self.sparse,
            )

        else:
            raise NotImplementedError(
                "Currently only works for nn.Linear, conv1d/conv2d and " "embeddingbag"
            )

    def save_activations(self, target, name):
        self.activation_save_target, self.activation_save_name = target, name
        # self.activation_save_target[self.activation_save_name] = []
        # self.activation_save_target[self.activation_save_name + '_Q'] = []

    def quantize_activations(self, activations, quantizer=None):
        """Quantize a single activation tensor or all activations from a layer. I'm assuming that
        we should quantize all outputs for a layer with the same quantization scheme.
        """
        # profiler.start("quantize_activations")
        if self.activation_function is not None:
            activations = self.activation_function(activations)

        if self.activation_save_target is not None:
            self.activation_save_target[self.activation_save_name] = (
                activations.data.cpu().numpy()
            )

        if self._quant_a:
            if quantizer is None:
                activations = self.activation_quantizer(activations)
            else:
                activations = quantizer(activations)

            if self.activation_save_target is not None:
                self.activation_save_target[self.activation_save_name + "_Q"] = (
                    activations.data.cpu().numpy()
                )
        # profiler.stop("quantize_activations")
        return activations
