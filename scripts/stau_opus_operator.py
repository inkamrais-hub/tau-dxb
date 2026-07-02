"""
τ-opus attention operator — fused forward/backward for training.

Replaces the monkey-patched eager_attention_forward with a custom autograd Function
that fuses the σ^τ normalization and both matmuls. Supports bf16/fp32, per-head
learnable τ and α, and causal/padding masks.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CLAMP_MIN = 1e-8


def _sigma_softplus(x): return F.softplus(x).clamp(min=CLAMP_MIN)
def _sigma_sigmoid(x):  return torch.sigmoid(x).clamp(min=CLAMP_MIN)
def _sigma_exp(x):      return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)
def _sigma_relu(x):     return F.relu(x).clamp(min=CLAMP_MIN)


_SIGMA_REGISTRY = {
    "softplus": _sigma_softplus,
    "sigmoid": _sigma_sigmoid,
    "exp": _sigma_exp,
    "relu": _sigma_relu,
}


def _sigma_prime_softplus(x): return torch.sigmoid(x)
def _sigma_prime_sigmoid(x):
    s = torch.sigmoid(x)
    return s * (1 - s)
def _sigma_prime_exp(x):      return torch.exp(x.clamp(max=15))
def _sigma_prime_relu(x):     return (x > 0).float()


_SIGMA_PRIME_REGISTRY = {
    "softplus": _sigma_prime_softplus,
    "sigmoid": _sigma_prime_sigmoid,
    "exp": _sigma_prime_exp,
    "relu": _sigma_prime_relu,
}


class STauOpusAttentionFunction(torch.autograd.Function):
    """
    Fused τ-opus attention: out = softmax_σ^τ(QK^T) V.

    Inputs:
        Q, K, V: (B, H, Lq, D), (B, H, Lk, D), (B, H, Lk, Dv)
        log_tau: (H,)   — tau = softplus(log_tau) + 1.0
        log_alpha: (H,) — alpha = exp(log_alpha)
        sigma_name: str
        attn_mask: optional (B, 1, Lq, Lk) or (B, H, Lq, Lk); -inf for masked.
        scale: float or None

    Outputs:
        out: (B, H, Lq, Dv)
        attn: (B, H, Lq, Lk)  — saved for analysis, not strictly needed for backward
    """

    @staticmethod
    def forward(ctx, Q, K, V, log_tau, log_alpha, sigma_name, attn_mask=None, scale=None):
        B, H, Lq, D = Q.shape
        _, _, Lk, _ = K.shape
        dtype = Q.dtype
        device = Q.device

        if scale is None:
            scale = D ** -0.5

        # Use fp32 for numerical stability of pow/log, unless input is fp64 (gradcheck)
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        Qf = Q.to(compute_dtype)
        Kf = K.to(compute_dtype)
        Vf = V.to(compute_dtype)

        scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * scale  # (B,H,Lq,Lk)
        if attn_mask is not None:
            scores = scores + attn_mask.to(compute_dtype)

        # max-stable shift
        s_max = scores.max(dim=-1, keepdim=True).values
        s_stable = scores - s_max

        alpha = torch.exp(log_alpha).view(1, H, 1, 1).to(device)
        tau = (F.softplus(log_tau) + 1.0).view(1, H, 1, 1).to(device)

        sigma_fn = _SIGMA_REGISTRY[sigma_name]
        sigma_val = sigma_fn(s_stable * alpha).clamp(min=CLAMP_MIN)
        q = sigma_val.pow(tau).clamp(min=CLAMP_MIN)
        Z = q.sum(dim=-1, keepdim=True).clamp(min=CLAMP_MIN)
        attn = q / Z

        out = torch.matmul(attn, Vf)  # (B,H,Lq,Dv)

        # Save minimal intermediates:
        #   - s_stable: needed for score backward + sigma recomputation + alpha grad
        #   - attn: needed for V grad + tau/alpha grad; recomputing costs pow
        #   - argmax (LongTensor): replaces full scores tensor (~36x smaller)
        #   - sigma_val is RECOMPUTED from s_stable in backward (saves one large tensor)
        ctx.save_for_backward(Qf, Kf, Vf, s_stable, attn, alpha, tau, log_tau, log_alpha,
                              scores.argmax(dim=-1, keepdim=True))
        ctx.sigma_name = sigma_name
        ctx.scale = scale

        return out.to(dtype), attn.to(dtype)

    @staticmethod
    def backward(ctx, grad_out, grad_attn_unused):
        Qf, Kf, Vf, s_stable, attn, alpha, tau, log_tau, log_alpha, argmax = ctx.saved_tensors
        sigma_name = ctx.sigma_name
        scale = ctx.scale
        B, H, Lq, D = Qf.shape
        _, _, Lk, Dv = Vf.shape

        compute_dtype = Qf.dtype
        grad_out_f = grad_out.to(compute_dtype)

        # Backward through out = attn @ V
        grad_attn = torch.matmul(grad_out_f, Vf.transpose(-2, -1))  # (B,H,Lq,Lk)
        grad_V = torch.matmul(attn.transpose(-2, -1), grad_out_f)   # (B,H,Lk,Dv)

        # Recompute sigma_val from s_stable (saves one (B,H,Lq,Lk) saved tensor)
        sigma_fn = _SIGMA_REGISTRY[sigma_name]
        sigma_val = sigma_fn(s_stable * alpha).clamp(min=CLAMP_MIN)

        # Backward through τ-opus normalization: attn_i = q_i / Z
        # d loss / d log(sigma_j) = tau * attn_j * (grad_attn_j - sum_i grad_attn_i * attn_i)
        weighted = (grad_attn * attn).sum(dim=-1, keepdim=True)  # (B,H,Lq,1)
        d_log_sigma = tau * attn * (grad_attn - weighted)        # (B,H,Lq,Lk)

        # Backward through log(sigma) w.r.t. s_stable * alpha
        sigma_prime_fn = _SIGMA_PRIME_REGISTRY.get(sigma_name, lambda x: torch.ones_like(x))
        sigma_p = sigma_prime_fn(s_alpha := s_stable * alpha).clamp(min=CLAMP_MIN)
        d_s_stable = d_log_sigma * (sigma_p / sigma_val) * alpha  # (B,H,Lq,Lk)

        # Backward through max-stable shift: dL/ds_k = dL/dy_k - sum(dL/dy) * indicator(k==argmax)
        d_scores = d_s_stable.clone()
        sum_dy = d_s_stable.sum(dim=-1, keepdim=True)
        d_scores.scatter_add_(-1, argmax, -sum_dy)

        # Backward through scores = Q @ K^T * scale
        grad_Q = torch.matmul(d_scores, Kf) * scale  # (B,H,Lq,D)
        grad_K = torch.matmul(d_scores.transpose(-2, -1), Qf) * scale  # (B,H,Lk,D)

        # Gradients w.r.t. log_tau — NO extra matmul needed, use grad_attn directly
        # dL/dτ_h = sum_{q,k} grad_attn[q,k] * d(attn[q,k])/dτ
        # d(attn_i)/dτ = attn_i * (log(sigma_i) - sum_j attn_j * log(sigma_j))
        log_sigma = sigma_val.log().clamp(min=-30)
        e_log_sigma = (attn * log_sigma).sum(dim=-1, keepdim=True)
        d_attn_dtau = attn * (log_sigma - e_log_sigma)  # (B,H,Lq,Lk)
        grad_tau_scalar = (grad_attn * d_attn_dtau).sum(dim=(0, 2, 3))  # (H,) — no matmul!
        d_tau_d_logtau = torch.sigmoid(log_tau)  # (H,)
        grad_log_tau = grad_tau_scalar * d_tau_d_logtau

        # Gradients w.r.t. log_alpha — same optimization
        d_logsigma_dalpha = (sigma_p / sigma_val * s_stable).clamp(-30, 30)  # (B,H,Lq,Lk)
        d_attn_dalpha = attn * tau * (d_logsigma_dalpha - (attn * d_logsigma_dalpha).sum(dim=-1, keepdim=True))
        grad_alpha_scalar = (grad_attn * d_attn_dalpha).sum(dim=(0, 2, 3))  # (H,) — no matmul!
        grad_log_alpha = grad_alpha_scalar * torch.exp(log_alpha)

        return (
            grad_Q.to(Qf.dtype), grad_K.to(Kf.dtype), grad_V.to(Vf.dtype),
            grad_log_tau, grad_log_alpha, None, None, None
        )


class STauOpusAttentionModule(nn.Module):
    """
    Drop-in replacement for a single attention head group.
    Not used directly inside Whisper; patch_whisper_stau.py uses the Function.
    """
    def __init__(self, n_heads, sigma_name="softplus"):
        super().__init__()
        self.n_heads = n_heads
        self.log_tau = nn.Parameter(torch.zeros(n_heads))
        self.log_alpha = nn.Parameter(torch.zeros(n_heads))
        self.sigma_name = sigma_name

    def forward(self, Q, K, V, attn_mask=None, scale=None):
        out, _ = STauOpusAttentionFunction.apply(
            Q, K, V, self.log_tau, self.log_alpha, self.sigma_name, attn_mask, scale
        )
        return out


def _log_tau_for_init(tau: float) -> float:
    if tau <= 1.001:
        return -8.0
    return math.log(math.exp(tau - 1.0) - 1.0)


def init_log_tau_from_values(tau_list):
    return torch.tensor([_log_tau_for_init(t) for t in tau_list], dtype=torch.float32)


if __name__ == "__main__":
    # Gradient self-test
    B, H, L, D = 2, 4, 16, 32
    Q = torch.randn(B, H, L, D, requires_grad=True)
    K = torch.randn(B, H, L, D, requires_grad=True)
    V = torch.randn(B, H, L, D, requires_grad=True)
    log_tau = nn.Parameter(torch.zeros(H))
    log_alpha = nn.Parameter(torch.zeros(H))

    out, attn = STauOpusAttentionFunction.apply(Q, K, V, log_tau, log_alpha, "softplus")
    loss = out.sum()
    loss.backward()

    print("Q grad:", Q.grad.shape, Q.grad.abs().mean().item())
    print("log_tau grad:", log_tau.grad.shape, log_tau.grad.abs().mean().item())
    print("log_alpha grad:", log_alpha.grad.shape, log_alpha.grad.abs().mean().item())
    print("OK")
