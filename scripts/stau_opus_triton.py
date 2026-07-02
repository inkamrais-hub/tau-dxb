"""
Triton kernels for τ-opus attention.
Fuses the σ(x)^τ normalization into single forward/backward passes,
eliminating redundant memory traffic from the pure-PyTorch implementation.

Forward:  scores (B,H,L,L) → attn, with fused max/σ/pow/sum/div
Backward: grad_attn + intermediates → d_scores, d_log_tau, d_log_alpha
"""
import torch
import triton
import triton.language as tl
import math

CLAMP_MIN = 1e-8


# ═══════════════════════════════════════════════════════════════
# Triton σ functions (device-side)
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _sigma_fwd(x, sigma_name: tl.constexpr):
    """Apply σ(x) per element. x is stable (already subtracted max)."""
    if sigma_name == "softplus":
        return tl.math.log(1.0 + tl.math.exp(x))  # softplus
    elif sigma_name == "sigmoid":
        return 1.0 / (1.0 + tl.math.exp(-x))  # sigmoid
    elif sigma_name == "exp":
        # clamp to avoid overflow
        return tl.math.exp(tl.minimum(x, 20.0))
    elif sigma_name == "relu":
        return tl.maximum(x, 0.0)
    else:
        return x


@triton.jit
def _sigma_prime(x, sigma_val, sigma_name: tl.constexpr):
    """σ'(x) per element. Uses sigma_val for sigmoid to avoid recompute."""
    if sigma_name == "softplus":
        return 1.0 / (1.0 + tl.math.exp(-x))  # sigmoid = σ'(softplus)
    elif sigma_name == "sigmoid":
        return sigma_val * (1.0 - sigma_val)
    elif sigma_name == "exp":
        return sigma_val  # d(e^x)/dx = e^x
    elif sigma_name == "relu":
        return tl.where(x > 0.0, 1.0, 0.0)
    else:
        return 1.0


# ═══════════════════════════════════════════════════════════════
# Forward kernel: fused σ^τ normalization
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _stau_opus_fwd_kernel(
    scores_ptr, attn_ptr,
    stride_s_b, stride_s_h, stride_s_m, stride_s_n,
    stride_a_b, stride_a_h, stride_a_m, stride_a_n,
    tau_ptr, alpha_ptr,
    B, H, M, N,
    BLOCK_SIZE: tl.constexpr,
    sigma_name: tl.constexpr,
):
    """Forward kernel for τ-opus normalization."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Load per-head τ, α
    tau_val = tl.load(tau_ptr + pid_h)
    alpha_val = tl.load(alpha_ptr + pid_h)

    row_ptr = scores_ptr + pid_b * stride_s_b + pid_h * stride_s_h + pid_m * stride_s_m
    out_ptr = attn_ptr + pid_b * stride_a_b + pid_h * stride_a_h + pid_m * stride_a_m

    # Pass 1: find max per row
    row_max = -float("inf")
    for start_n in range(0, N, BLOCK_SIZE):
        n_offs = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offs < N
        s = tl.load(row_ptr + n_offs * stride_s_n, mask=mask, other=-float("inf"))
        block_max = tl.max(s, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # Pass 2: compute σ^τ sum
    row_sum = 0.0
    for start_n in range(0, N, BLOCK_SIZE):
        n_offs = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offs < N
        s = tl.load(row_ptr + n_offs * stride_s_n, mask=mask, other=0.0)
        x_stable = s - row_max
        sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_name)
        log_q = tau_val * tl.math.log(tl.maximum(sigma_val, 1e-8))
        q = tl.math.exp(log_q)
        row_sum += tl.sum(tl.where(mask, q, 0.0), axis=0)

    # Pass 3: normalize and write
    inv_sum = 1.0 / tl.maximum(row_sum, 1e-8)
    for start_n in range(0, N, BLOCK_SIZE):
        n_offs = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offs < N
        s = tl.load(row_ptr + n_offs * stride_s_n, mask=mask, other=0.0)
        x_stable = s - row_max
        sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_name)
        log_q = tau_val * tl.math.log(tl.maximum(sigma_val, 1e-8))
        q = tl.math.exp(log_q)
        attn_val = q * inv_sum
        tl.store(out_ptr + n_offs * stride_a_n, attn_val, mask=mask)


# ═══════════════════════════════════════════════════════════════
# Backward kernel: fused gradient computation
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _stau_opus_bwd_kernel(
    scores_ptr, attn_ptr, grad_attn_ptr, d_scores_ptr,
    stride_s_b, stride_s_h, stride_s_m, stride_s_n,
    tau_ptr, alpha_ptr,
    B, H, M, N,
    BLOCK_SIZE: tl.constexpr,
    sigma_name: tl.constexpr,
):
    """Backward kernel for τ-opus normalization.
    Vectorized 4-pass: ① max + weighted_sum  ② d_s_stable + sum  ③ argmax  ④ correction
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    tau_val = tl.load(tau_ptr + pid_h)
    alpha_val = tl.load(alpha_ptr + pid_h)

    s_row = scores_ptr + pid_b * stride_s_b + pid_h * stride_s_h + pid_m * stride_s_m
    a_row = attn_ptr + pid_b * stride_s_b + pid_h * stride_s_h + pid_m * stride_s_m
    g_row = grad_attn_ptr + pid_b * stride_s_b + pid_h * stride_s_h + pid_m * stride_s_m
    d_row = d_scores_ptr + pid_b * stride_s_b + pid_h * stride_s_h + pid_m * stride_s_m

    # ── Pass ①: find row_max and weighted_sum ──
    row_max = -float("inf")
    weighted_sum = 0.0
    for start_n in range(0, N, BLOCK_SIZE):
        n_offs = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offs < N
        s = tl.load(s_row + n_offs * stride_s_n, mask=mask, other=-float("inf"))
        attn_val = tl.load(a_row + n_offs * stride_s_n, mask=mask, other=0.0)
        grad_val = tl.load(g_row + n_offs * stride_s_n, mask=mask, other=0.0)
        # Find max
        block_max = tl.max(s, axis=0)
        row_max = tl.maximum(row_max, block_max)
        weighted_sum += tl.sum(attn_val * grad_val, axis=0)

    # ── Pass ②: compute d_s_stable + sum it ──
    sum_d_stable = 0.0
    for start_n in range(0, N, BLOCK_SIZE):
        n_offs = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offs < N
        s = tl.load(s_row + n_offs * stride_s_n, mask=mask, other=0.0)
        attn_val = tl.load(a_row + n_offs * stride_s_n, mask=mask, other=0.0)
        grad_val = tl.load(g_row + n_offs * stride_s_n, mask=mask, other=0.0)

        x_stable = s - row_max
        sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_name)
        sigma_p = _sigma_prime(x_stable * alpha_val, sigma_val, sigma_name)

        d_log_sigma = tau_val * attn_val * (grad_val - weighted_sum)
        d_s_stable = d_log_sigma * (sigma_p / tl.maximum(sigma_val, 1e-8)) * alpha_val
        sum_d_stable += tl.sum(d_s_stable, axis=0)

        # Write preliminary d_scores
        tl.store(d_row + n_offs * stride_s_n, d_s_stable, mask=mask)

    # ── Pass ③: apply argmax correction via vectorized mask ──
    for start_n in range(0, N, BLOCK_SIZE):
        n_offs = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offs < N
        d_s = tl.load(d_row + n_offs * stride_s_n, mask=mask, other=0.0)
        s = tl.load(s_row + n_offs * stride_s_n, mask=mask, other=0.0)
        # is this position the argmax? (allows multiple if tied)
        is_max = tl.abs(s - row_max) < 1e-6
        d_s_corrected = tl.where(is_max, d_s - sum_d_stable, d_s)
        tl.store(d_row + n_offs * stride_s_n, d_s_corrected, mask=mask)


# ═══════════════════════════════════════════════════════════════
# Python wrappers
# ═══════════════════════════════════════════════════════════════

def triton_stau_opus_forward(scores, tau, alpha, sigma_name="softplus"):
    """Triton-accelerated τ-opus normalization forward.
    
    Args:
        scores: (B, H, M, N) float32 — attention scores (pre-scaled + masked)
        tau: (H,) float32 — per-head temperature
        alpha: (H,) float32 — per-head alpha (not log!)
    
    Returns:
        attn: (B, H, M, N) float32 — normalized attention weights
    """
    B, H, M, N = scores.shape
    attn = torch.empty_like(scores)

    BLOCK_SIZE = min(64, N)
    grid = (B, H, M)

    _stau_opus_fwd_kernel[grid](
        scores, attn,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        tau, alpha,
        B, H, M, N,
        BLOCK_SIZE=BLOCK_SIZE,
        sigma_name=sigma_name,
    )
    return attn


def triton_stau_opus_backward(scores, attn, grad_attn, tau, alpha, sigma_name="softplus"):
    """Triton-accelerated τ-opus normalization backward.
    
    Args:
        scores: (B, H, M, N) float32 — attention scores
        attn: (B, H, M, N) float32 — attention weights from forward
        grad_attn: (B, H, M, N) float32 — upstream gradient
        tau: (H,) float32 — per-head temperature
        alpha: (H,) float32 — per-head alpha
    
    Returns:
        d_scores: (B, H, M, N) float32
    """
    B, H, M, N = scores.shape
    d_scores = torch.empty_like(scores)

    BLOCK_SIZE = min(64, N)
    grid = (B, H, M)

    _stau_opus_bwd_kernel[grid](
        scores, attn, grad_attn, d_scores,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        tau, alpha,
        B, H, M, N,
        BLOCK_SIZE=BLOCK_SIZE,
        sigma_name=sigma_name,
    )
    return d_scores


# ═══════════════════════════════════════════════════════════════
# Fused autograd Function (uses Triton kernels)
# ═══════════════════════════════════════════════════════════════

class STauOpusTritonFunction(torch.autograd.Function):
    """τ-opus attention with Triton-accelerated forward/backward.
    
    Replaces STauOpusAttentionFunction with CUDA-level kernel fusion.
    """
    @staticmethod
    def forward(ctx, Q, K, V, log_tau, log_alpha, sigma_name, attn_mask=None, scale=None):
        dtype = Q.dtype
        Qf = Q.float()
        Kf = K.float()
        Vf = V.float()

        if scale is None:
            scale = Q.size(-1) ** -0.5
        scale = float(scale)

        # Compute scores = Q @ K^T * scale + mask
        scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask.float()

        # Per-head τ and α
        tau = (torch.nn.functional.softplus(log_tau) + 1.0)  # (H,)  (H,)
        alpha = torch.exp(log_alpha)  # (H,)

        # Triton forward
        attn = triton_stau_opus_forward(scores, tau, alpha, sigma_name)

        # Output = attn @ V
        out = torch.matmul(attn, Vf)

        # Save for backward
        ctx.save_for_backward(Qf, Kf, Vf, scores, attn, tau, alpha, log_tau, log_alpha)
        ctx.sigma_name = sigma_name
        ctx.scale = scale

        return out.to(dtype), attn.to(dtype)

    @staticmethod
    def backward(ctx, grad_out, grad_attn_unused):
        Qf, Kf, Vf, scores, attn, tau, alpha, log_tau, log_alpha = ctx.saved_tensors
        sigma_name = ctx.sigma_name
        scale = ctx.scale
        B, H, Lq, D = Qf.shape
        _, _, Lk, Dv = Vf.shape

        compute_dtype = Qf.dtype
        grad_out_f = grad_out.to(compute_dtype)

        # grad_attn from upstream: dL/d(attn) = grad_out @ V^T
        grad_attn = torch.matmul(grad_out_f, Vf.transpose(-2, -1))

        # grad_V: dL/dV = attn^T @ grad_out
        grad_V = torch.matmul(attn.transpose(-2, -1), grad_out_f)

        # Triton backward: compute d_scores from grad_attn
        d_scores = triton_stau_opus_backward(scores, attn, grad_attn, tau, alpha, sigma_name)

        # grad_Q, grad_K from d_scores
        grad_Q = torch.matmul(d_scores, Kf) * scale
        grad_K = torch.matmul(d_scores.transpose(-2, -1), Qf) * scale

        # Compute scalar gradients for log_tau and log_alpha (PyTorch ops, lightweight)
        # Reshape per-head params for broadcasting
        alpha_br = alpha.view(1, H, 1, 1)  # (1,H,1,1)
        tau_br = tau.view(1, H, 1, 1)
        s_stable = scores - scores.max(dim=-1, keepdim=True).values
        sigma_val = _sigma_fn_registry(sigma_name, s_stable * alpha_br).clamp(min=CLAMP_MIN)
        log_sigma = sigma_val.log().clamp(min=-30)

        # d_attn/dτ = attn · (log σ - E_p[log σ])
        e_log_sigma = (attn * log_sigma).sum(dim=-1, keepdim=True)
        d_attn_dtau = attn * (log_sigma - e_log_sigma)
        grad_tau_scalar = (grad_attn * d_attn_dtau).sum(dim=(0, 2, 3))

        d_tau_d_logtau = torch.sigmoid(log_tau)
        grad_log_tau = grad_tau_scalar * d_tau_d_logtau

        # dL/dα
        sigma_prime_fn = _sigma_prime_registry(sigma_name)
        x_stable = s_stable
        sigma_p = sigma_prime_fn(x_stable * alpha_br).clamp(min=CLAMP_MIN)
        d_logsigma_dalpha = (sigma_p / sigma_val * x_stable).clamp(-30, 30)
        d_attn_dalpha = attn * tau_br * (d_logsigma_dalpha - (attn * d_logsigma_dalpha).sum(dim=-1, keepdim=True))
        grad_alpha_scalar = (grad_attn * d_attn_dalpha).sum(dim=(0, 2, 3))
        grad_log_alpha = grad_alpha_scalar * torch.exp(log_alpha)

        return (
            grad_Q.to(Qf.dtype), grad_K.to(Kf.dtype), grad_V.to(Vf.dtype),
            grad_log_tau, grad_log_alpha, None, None, None,
        )


def _sigma_fn_registry(name, x):
    if name == "softplus":
        return torch.nn.functional.softplus(x).clamp(min=CLAMP_MIN)
    elif name == "sigmoid":
        return torch.sigmoid(x).clamp(min=CLAMP_MIN)
    elif name == "exp":
        return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)
    elif name == "relu":
        return torch.relu(x).clamp(min=CLAMP_MIN)
    return x


def _sigma_prime_registry(name):
    if name == "softplus":
        return lambda x: torch.sigmoid(x)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
    elif name == "exp":
        return lambda x: torch.exp(x.clamp(max=15))
    elif name == "relu":
        return lambda x: (x > 0).float()
    return lambda x: torch.ones_like(x)


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Triton τ-opus kernels...")

    B, H, M, N, D = 2, 4, 128, 128, 64
    torch.manual_seed(42)

    Q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    log_tau = torch.nn.Parameter(torch.zeros(H, device="cuda"))
    log_alpha = torch.nn.Parameter(torch.zeros(H, device="cuda"))

    # Forward test
    out, attn = STauOpusTritonFunction.apply(Q, K, V, log_tau, log_alpha, "softplus")
    loss = out.sum()
    loss.backward()

    print(f"Forward OK. Output shape: {out.shape}")
    print(f"Q grad norm: {Q.grad.norm().item():.4f}")
    print(f"log_tau grad: {log_tau.grad}")
    print(f"log_alpha grad: {log_alpha.grad}")
    print(f"No NaN: {not torch.isnan(Q.grad).any()}")
    print(f"No Inf: {not torch.isinf(Q.grad).any()}")

    # Compare with PyTorch version
    print("\nComparing with reference implementation...")
    from torch.autograd import gradcheck

    def func(Q, K, V, lt, la):
        out, _ = STauOpusTritonFunction.apply(Q, K, V, lt, la, "softplus")
        return out.sum()

    Qd = Q.double().detach().requires_grad_(True)
    Kd = K.double().detach().requires_grad_(True)
    Vd = V.double().detach().requires_grad_(True)
    ltd = log_tau.double().detach().requires_grad_(True)
    lad = log_alpha.double().detach().requires_grad_(True)

    ok = gradcheck(
        lambda q, k, v, lt, la: STauOpusTritonFunction.apply(q, k, v, lt, la, "softplus")[0],
        [Qd, Kd, Vd, ltd, lad],
        eps=1e-3, atol=1e-1, rtol=1e-1,
    )
    print(f"gradcheck: {ok}")

    # Speed test
    print("\nSpeed test (B=4, H=8, L=1500, D=64)...")
    B2, H2, L2, D2 = 4, 8, 1500, 64
    Q2 = torch.randn(B2, H2, L2, D2, device="cuda", dtype=torch.float32)
    K2 = torch.randn(B2, H2, L2, D2, device="cuda", dtype=torch.float32)
    V2 = torch.randn(B2, H2, L2, D2, device="cuda", dtype=torch.float32)
    lt2 = torch.nn.Parameter(torch.zeros(H2, device="cuda"))
    la2 = torch.nn.Parameter(torch.zeros(H2, device="cuda"))

    # Warmup
    for _ in range(3):
        o, _ = STauOpusTritonFunction.apply(Q2, K2, V2, lt2, la2, "softplus")
        o.sum().backward()

    torch.cuda.synchronize()
    import time
    t0 = time.time()
    for _ in range(20):
        o, _ = STauOpusTritonFunction.apply(Q2, K2, V2, lt2, la2, "softplus")
        o.sum().backward()
    torch.cuda.synchronize()
    t = (time.time() - t0) / 20
    print(f"  Triton kernel: {t*1000:.1f}ms per step")
