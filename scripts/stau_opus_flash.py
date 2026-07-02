"""
FlashAttention-style τ-opus kernel — single fused Triton kernel for
Q@K^T → σ^τ normalization → attn@V (forward + backward).

Avoids materializing the (B, H, L, L) score matrix to HBM.
Stores (m, l, argmax_idx) normalization constants for backward.

FIXED dQ/dK bug: argmax correction now uses GLOBAL sum of F across all
KV blocks, not per-block local sum.
"""
import torch
import triton
import triton.language as tl

CLAMP_MIN = 1e-8


# ═══════════════════════════════════════════════════════════════════
# Device-side σ functions (constexpr dispatch)
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _sigma_fwd(x, sigma_name: tl.constexpr):
    if sigma_name == "softplus":
        return tl.math.log(1.0 + tl.math.exp(tl.minimum(x, 30.0)))
    elif sigma_name == "sigmoid":
        return 1.0 / (1.0 + tl.math.exp(-tl.minimum(x, 30.0)))
    elif sigma_name == "exp":
        return tl.math.exp(tl.minimum(x, 20.0))
    elif sigma_name == "relu":
        return tl.maximum(x, 0.0)
    return x


@triton.jit
def _sigma_prime_fn(x, sigma_val, sigma_name: tl.constexpr):
    if sigma_name == "softplus":
        return 1.0 / (1.0 + tl.math.exp(-tl.minimum(x, 30.0)))  # sigmoid
    elif sigma_name == "sigmoid":
        return sigma_val * (1.0 - sigma_val)
    elif sigma_name == "exp":
        return sigma_val
    elif sigma_name == "relu":
        return tl.where(x > 0.0, 1.0, 0.0)
    return tl.zeros_like(x) + 1.0


# ═══════════════════════════════════════════════════════════════════
# Helper: compute d_s (uncorrected) and attn for a score tile
# Returns: attn (BLOCK_M, BLOCK_N), dP (BLOCK_M, BLOCK_N),
#          F (uncorrected d_s, BLOCK_M, BLOCK_N)
#          d_tau_contrib (BLOCK_M,), d_alpha_contrib (BLOCK_M,)
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _compute_attn_and_F(
    Q_tile, K_tile, V_tile, dO_tile,
    m_row, inv_l,
    tau_val, alpha_val,
    sigma_name: tl.constexpr,
):
    """Recompute attention and compute dP, F (uncorrected score gradient).

    Returns:
        attn, dP, F, d_tau_contrib, d_alpha_contrib
    """
    # s = Q @ K^T
    s = tl.dot(Q_tile, K_tile.T)  # (BLOCK_M, BLOCK_N)

    # σ^τ normalization → attn
    x_stable = s - m_row[:, None]
    sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_name)
    log_q = tau_val * tl.math.log(tl.maximum(sigma_val, 1e-8))
    q = tl.math.exp(log_q)
    attn = q * inv_l[:, None]  # (BLOCK_M, BLOCK_N)

    # dP = dO @ V^T
    dP = tl.dot(dO_tile, V_tile.T)  # (BLOCK_M, BLOCK_N)

    # weighted = Σ P · dP  (per query row)
    weighted = tl.sum(attn * dP, axis=1)  # (BLOCK_M,)

    # F = τ · P · (dP - E[P·dP]) · σ'/σ · α  (uncorrected score gradient)
    d_log_sigma = tau_val * attn * (dP - weighted[:, None])
    sigma_p = _sigma_prime_fn(x_stable * alpha_val, sigma_val, sigma_name)
    F = d_log_sigma * (sigma_p / tl.maximum(sigma_val, 1e-8)) * alpha_val

    # dτ buffer: Σ attn · dP · (log σ - E[log σ])
    log_sigma_val = tl.math.log(tl.maximum(sigma_val, 1e-8))
    e_log_sigma = tl.sum(attn * log_sigma_val, axis=1)
    d_attn_dtau = attn * (log_sigma_val - e_log_sigma[:, None])
    d_tau_contrib = tl.sum(dP * d_attn_dtau, axis=1)

    # dα buffer
    inner_dalpha = (sigma_p / tl.maximum(sigma_val, 1e-8)) * x_stable
    e_inner = tl.sum(attn * inner_dalpha, axis=1)
    d_attn_dalpha = attn * tau_val * (inner_dalpha - e_inner[:, None])
    d_alpha_contrib = tl.sum(dP * d_attn_dalpha, axis=1)

    return attn, dP, F, d_tau_contrib, d_alpha_contrib


# ═══════════════════════════════════════════════════════════════════
# Forward kernel — tiled QK^T → σ^τ → attn@V
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _stau_opus_flash_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    m_ptr, l_ptr, argmax_ptr,  # (B, H, Lq) stored for backward
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_mb, stride_mh, stride_mm,
    stride_lb, stride_lh, stride_lm,
    stride_ab, stride_ah, stride_am,
    tau_ptr, alpha_ptr,
    B, H, Lq, Lk, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    sigma_name: tl.constexpr,
):
    """Fused τ-opus attention forward.

    Grid: (B, H, num_q_blocks)
    Each program handles BLOCK_M query rows for one (B,H) pair.

    Stores m (row max), l (row sum of σ^τ), argmax_idx for backward.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)  # Q-block index

    num_m_blocks = tl.cdiv(Lq, BLOCK_M)
    if pid_m >= num_m_blocks:
        return

    q_start = pid_m * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offs < Lq

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Per-head params
    tau_val = tl.load(tau_ptr + pid_h)
    alpha_val = tl.load(alpha_ptr + pid_h)

    # ── Load Q tile ──
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_D)

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum of σ^τ
    O_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # accumulator
    argmax_i = tl.zeros([BLOCK_M], dtype=tl.int32) - 1  # argmax index

    # ── Pass 1 over KV: find row max ──
    for start_n in range(0, Lk, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < Lk

        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        s = tl.dot(Q_tile, K_tile.T, input_precision="ieee")
        s = tl.where(n_mask[None, :], s, -float('inf'))

        block_max = tl.max(s, axis=1)
        # Track argmax: update when we find a new max
        new_max = block_max > m_i
        # For each Q row, the argmax position = start_n + argmax position in this block
        block_argmax = tl.argmax(s, axis=1)  # (BLOCK_M,), index within block
        argmax_i = tl.where(new_max, start_n + block_argmax, argmax_i)
        m_i = tl.maximum(m_i, block_max)

    # ── Pass 2 over KV: compute σ^τ and accumulate O ──
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    O_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, Lk, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < Lk

        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        s = tl.dot(Q_tile, K_tile.T, input_precision="ieee")
        s = tl.where(n_mask[None, :], s, -float('inf'))

        x_stable = s - m_i[:, None]
        sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_name)
        log_q = tau_val * tl.math.log(tl.maximum(sigma_val, 1e-8))
        q = tl.math.exp(log_q)

        l_block = tl.sum(q, axis=1)
        l_i += l_block
        O_i += tl.dot(q.to(tl.float32), V_tile, input_precision="ieee")

    # ── Normalize ──
    inv_l = 1.0 / tl.maximum(l_i, 1e-8)
    O_i = O_i * inv_l[:, None]

    # ── Write output ──
    o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_od
    tl.store(o_ptrs, O_i.to(Q_tile.dtype), mask=q_mask[:, None] & d_mask[None, :])

    # ── Write m, l, argmax for backward ──
    m_ptrs = m_ptr + pid_b * stride_mb + pid_h * stride_mh + q_offs * stride_mm
    l_ptrs = l_ptr + pid_b * stride_lb + pid_h * stride_lh + q_offs * stride_lm
    a_ptrs = argmax_ptr + pid_b * stride_ab + pid_h * stride_ah + q_offs * stride_am
    tl.store(m_ptrs, m_i, mask=q_mask)
    tl.store(l_ptrs, l_i, mask=q_mask)
    tl.store(a_ptrs, argmax_i, mask=q_mask)


# ═══════════════════════════════════════════════════════════════════
# Backward kernel — dQ (iterate over KV blocks)
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _stau_opus_flash_bwd_dq(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, dQ_ptr,
    m_ptr, l_ptr, total_F_ptr,  # total_F_ptr: per-Q-row sum of F (output)
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_mb, stride_mh, stride_mm,
    stride_lb, stride_lh, stride_lm,
    stride_fb, stride_fh, stride_fm,
    tau_ptr, alpha_ptr,
    B, H, Lq, Lk, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    sigma_name: tl.constexpr,
):
    """Backward dQ kernel — for each Q-block, iterate over KV and accumulate dQ.

    dQ_block = Σ F_i @ K_i  (NO per-block argmax correction)
    total_F_j = Σ F_ji (global sum, stored for post-processing)

    Uses O (forward output) to compute global weighted = dO·O,
    eliminating the per-block weighted approximation error.

    The argmax correction is applied in PyTorch after both kernels:
        dQ[j,:] -= total_F_j * K[argmax_j,:]
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    num_m_blocks = tl.cdiv(Lq, BLOCK_M)
    if pid_m >= num_m_blocks:
        return

    q_start = pid_m * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offs < Lq

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    tau_val = tl.load(tau_ptr + pid_h)
    alpha_val = tl.load(alpha_ptr + pid_h)

    # Load Q, dO, O (forward output), m, l
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    do_ptrs = dO_ptr + pid_b * stride_dob + pid_h * stride_doh + q_offs[:, None] * stride_dom + d_offs[None, :] * stride_dod
    dO_tile = tl.load(do_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_od
    O_tile = tl.load(o_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    m_row = tl.load(m_ptr + pid_b * stride_mb + pid_h * stride_mh + q_offs * stride_mm, mask=q_mask, other=0.0)
    l_row = tl.load(l_ptr + pid_b * stride_lb + pid_h * stride_lh + q_offs * stride_lm, mask=q_mask, other=0.0)

    inv_l = 1.0 / tl.maximum(l_row, 1e-8)

    # Global weighted = dO · O  (exact, no per-block approximation)
    weighted_global = tl.sum(dO_tile * O_tile, axis=1)  # (BLOCK_M,)

    # Also compute dtau and dalpha for this Q-block
    d_tau_buffer = tl.zeros([BLOCK_M], dtype=tl.float32)
    d_alpha_buffer = tl.zeros([BLOCK_M], dtype=tl.float32)
    dQ_block = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    total_F = tl.zeros([BLOCK_M], dtype=tl.float32)  # global sum of F per Q row

    for start_n in range(0, Lk, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < Lk

        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Mask for KV positions
        s = tl.dot(Q_tile, K_tile.T, input_precision="ieee")
        s_masked = tl.where(n_mask[None, :], s, -float('inf'))

        # Recompute attn, F (uncorrected score gradient)
        x_stable = s_masked - m_row[:, None]
        sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_name)
        log_q = tau_val * tl.math.log(tl.maximum(sigma_val, 1e-8))
        q = tl.math.exp(log_q)
        attn = q * inv_l[:, None]

        dP = tl.dot(dO_tile, V_tile.T, input_precision="ieee")

        # Use GLOBAL weighted (dO·O) instead of per-block weighted
        d_log_sigma = tau_val * attn * (dP - weighted_global[:, None])
        sigma_p = _sigma_prime_fn(x_stable * alpha_val, sigma_val, sigma_name)
        F = d_log_sigma * (sigma_p / tl.maximum(sigma_val, 1e-8)) * alpha_val

        # Accumulate dQ_block += F @ K (NO argmax correction for individual tiles)
        dQ_block += tl.dot(F.to(tl.float32), K_tile, input_precision="ieee")

        # Accumulate global F sum for this Q-block
        total_F += tl.sum(F, axis=1)

        # dτ / dα contributions (need attn, attn·logσ, attn·inner_dα per block)
        log_sigma_val = tl.math.log(tl.maximum(sigma_val, 1e-8))
        e_log_sigma = tl.sum(attn * log_sigma_val, axis=1)
        d_attn_dtau = attn * (log_sigma_val - e_log_sigma[:, None])
        d_tau_buffer += tl.sum(dP * d_attn_dtau, axis=1)

        inner_dalpha = (sigma_p / tl.maximum(sigma_val, 1e-8)) * x_stable
        e_inner = tl.sum(attn * inner_dalpha, axis=1)
        d_attn_dalpha = attn * tau_val * (inner_dalpha - e_inner[:, None])
        d_alpha_buffer += tl.sum(dP * d_attn_dalpha, axis=1)

    # ── Write dQ (Σ F_i @ K_i, NO argmax correction) ──
    dq_ptrs = dQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh + q_offs[:, None] * stride_dqm + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, dQ_block.to(Q_tile.dtype), mask=q_mask[:, None] & d_mask[None, :])

    # ── Write total_F for post-processing ──
    tf_ptrs = total_F_ptr + pid_b * stride_fb + pid_h * stride_fh + q_offs * stride_fm
    tl.store(tf_ptrs, total_F, mask=q_mask)


# ═══════════════════════════════════════════════════════════════════
# Backward kernel — dK, dV (iterate over Q-blocks)
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _stau_opus_flash_bwd_dkdv(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, dK_ptr, dV_ptr,
    m_ptr, l_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dkb, stride_dkh, stride_dkm, stride_dkd,
    stride_dvb, stride_dvh, stride_dvm, stride_dvd,
    stride_mb, stride_mh, stride_mm,
    stride_lb, stride_lh, stride_lm,
    tau_ptr, alpha_ptr,
    B, H, Lq, Lk, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    sigma_name: tl.constexpr,
):
    """Backward dK/dV kernel — for each KV-block, iterate over Q-blocks.

    dK += F^T @ Q  (NO per-block argmax correction)
    dV += attn^T @ dO

    Uses O (forward output) to compute global weighted = dO·O,
    eliminating the per-block weighted approximation error.

    The argmax correction for dK is applied in PyTorch:
        dK[argmax_j,:] -= total_F_j * Q[j,:]
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)  # KV-block index

    num_n_blocks = tl.cdiv(Lk, BLOCK_N)
    if pid_n >= num_n_blocks:
        return

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < Lk

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    tau_val = tl.load(tau_ptr + pid_h)
    alpha_val = tl.load(alpha_ptr + pid_h)

    # Load K, V tiles (fixed for this KV-block)
    k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
    V_tile = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    dK_block = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dV_block = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    for start_m in range(0, Lq, BLOCK_M):
        q_offs = start_m + tl.arange(0, BLOCK_M)
        q_mask = q_offs < Lq

        q_ptr_i = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
        Q_tile = tl.load(q_ptr_i, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        do_ptr_i = dO_ptr + pid_b * stride_dob + pid_h * stride_doh + q_offs[:, None] * stride_dom + d_offs[None, :] * stride_dod
        dO_tile = tl.load(do_ptr_i, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Load O (forward output) for this Q-block
        o_ptr_i = O_ptr + pid_b * stride_ob + pid_h * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_od
        O_tile = tl.load(o_ptr_i, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        m_row = tl.load(m_ptr + pid_b * stride_mb + pid_h * stride_mh + q_offs * stride_mm, mask=q_mask, other=0.0)
        l_row = tl.load(l_ptr + pid_b * stride_lb + pid_h * stride_lh + q_offs * stride_lm, mask=q_mask, other=0.0)
        inv_l = 1.0 / tl.maximum(l_row, 1e-8)

        # Global weighted = dO · O  (exact)
        weighted_global = tl.sum(dO_tile * O_tile, axis=1)  # (BLOCK_M,)

        # s = Q @ K^T
        s = tl.dot(Q_tile, K_tile.T, input_precision="ieee")
        s_masked = tl.where(n_mask[None, :], s, -float('inf'))

        # Recompute attn and F (uncorrected)
        x_stable = s_masked - m_row[:, None]
        sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_name)
        log_q = tau_val * tl.math.log(tl.maximum(sigma_val, 1e-8))
        q = tl.math.exp(log_q)
        attn = q * inv_l[:, None]

        # dV += attn^T @ dO  (NO argmax correction needed for dV)
        dV_block += tl.dot(attn.T.to(tl.float32), dO_tile, input_precision="ieee")

        # dP and F (uncorrected score gradient) — using GLOBAL weighted
        dP = tl.dot(dO_tile, V_tile.T, input_precision="ieee")
        d_log_sigma = tau_val * attn * (dP - weighted_global[:, None])

        sigma_p = _sigma_prime_fn(x_stable * alpha_val, sigma_val, sigma_name)
        F = d_log_sigma * (sigma_p / tl.maximum(sigma_val, 1e-8)) * alpha_val

        # dK += F^T @ Q  (NO per-block argmax correction)
        dK_block += tl.dot(F.T.to(tl.float32), Q_tile, input_precision="ieee")

    # ── Write dK, dV ──
    dk_ptrs = dK_ptr + pid_b * stride_dkb + pid_h * stride_dkh + n_offs[:, None] * stride_dkm + d_offs[None, :] * stride_dkd
    dv_ptrs = dV_ptr + pid_b * stride_dvb + pid_h * stride_dvh + n_offs[:, None] * stride_dvm + d_offs[None, :] * stride_dvd
    tl.store(dk_ptrs, dK_block.to(K_tile.dtype), mask=n_mask[:, None] & d_mask[None, :])
    tl.store(dv_ptrs, dV_block.to(V_tile.dtype), mask=n_mask[:, None] & d_mask[None, :])


# ═══════════════════════════════════════════════════════════════════
# Python wrapper — autograd Function
# ═══════════════════════════════════════════════════════════════════

class STauOpusFlashFunction(torch.autograd.Function):
    """FlashAttention-style τ-opus with fused Triton kernels."""

    @staticmethod
    def forward(ctx, Q, K, V, log_tau, log_alpha, sigma_name, attn_mask=None, scale=None):
        dtype = Q.dtype
        Qf = Q.float()
        Kf = K.float()
        Vf = V.float()

        if scale is None:
            scale = Q.size(-1) ** -0.5
        scale = float(scale)

        B, H, Lq, D = Qf.shape
        _, _, Lk, _ = Kf.shape
        Vf = Vf.expand_as(Kf) if Vf.shape != Kf.shape else Vf

        # Apply scale to Q/K to absorb the factor
        Qs = Qf * (scale ** 0.5)
        Ks = Kf * (scale ** 0.5)

        # Output buffer (based on Q length)
        O = torch.empty(B, H, Lq, D, device='cuda', dtype=torch.float32)

        # Normalization constant buffers for backward
        m = torch.empty(B, H, Lq, device='cuda', dtype=torch.float32)
        l = torch.empty(B, H, Lq, device='cuda', dtype=torch.float32)
        argmax = torch.empty(B, H, Lq, device='cuda', dtype=torch.int32)

        # Per-head tensors
        tau = (torch.nn.functional.softplus(log_tau) + 1.0).float()
        alpha = torch.exp(log_alpha).float()

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_D = D
        num_m = triton.cdiv(Lq, BLOCK_M)

        grid = (B, H, num_m)

        _stau_opus_flash_fwd[grid](
            Qs, Ks, Vf, O, m, l, argmax,
            Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
            Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
            Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            m.stride(0), m.stride(1), m.stride(2),
            l.stride(0), l.stride(1), l.stride(2),
            argmax.stride(0), argmax.stride(1), argmax.stride(2),
            tau, alpha,
            B, H, Lq, Lk, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            sigma_name=sigma_name if isinstance(sigma_name, str) else "softplus",
        )

        ctx.save_for_backward(Qs, Ks, Vf, O, m, l, argmax, tau, alpha, log_tau, log_alpha)
        ctx.sigma_name = sigma_name
        ctx.scale = scale

        return O.to(dtype), m

    @staticmethod
    def backward(ctx, grad_out, _):
        Qs, Ks, Vf, O, m, l, argmax, tau, alpha, log_tau, log_alpha = ctx.saved_tensors
        sigma_name = ctx.sigma_name
        scale = ctx.scale
        B, H, Lq, D = Qs.shape
        _, _, Lk, _ = Ks.shape

        # δO gradient
        dO = grad_out.to(torch.float32)

        # Allocate gradient buffers
        dQ = torch.empty_like(Qs)
        dK = torch.empty(B, H, Lk, D, device='cuda', dtype=torch.float32)
        dV = torch.empty(B, H, Lk, D, device='cuda', dtype=torch.float32)
        total_F = torch.empty(B, H, Lq, device='cuda', dtype=torch.float32)  # per-Q-row ΣF

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_D = D
        num_m = triton.cdiv(Lq, BLOCK_M)
        num_n = triton.cdiv(Lk, BLOCK_N)

        # ── dQ kernel ──
        grid_dq = (B, H, num_m)
        _stau_opus_flash_bwd_dq[grid_dq](
            Qs, Ks, Vf, dO, O, dQ,
            m, l, total_F,
            Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
            Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
            Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            m.stride(0), m.stride(1), m.stride(2),
            l.stride(0), l.stride(1), l.stride(2),
            total_F.stride(0), total_F.stride(1), total_F.stride(2),
            tau, alpha,
            B, H, Lq, Lk, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            sigma_name=sigma_name if isinstance(sigma_name, str) else "softplus",
        )

        # ── dK, dV kernel ──
        grid_dkdv = (B, H, num_n)
        _stau_opus_flash_bwd_dkdv[grid_dkdv](
            Qs, Ks, Vf, dO, O, dK, dV,
            m, l,
            Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
            Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
            Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            m.stride(0), m.stride(1), m.stride(2),
            l.stride(0), l.stride(1), l.stride(2),
            tau, alpha,
            B, H, Lq, Lk, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            sigma_name=sigma_name if isinstance(sigma_name, str) else "softplus",
        )

        # ── Global argmax correction (PyTorch) ──
        # dQ[j,:] -= total_F[j] * Ks[argmax_j,:]
        # Expand index to gather Ks: (B, H, Lq) → (B, H, Lq, D)
        argmax_i64 = argmax.long()  # gather needs int64
        argmax_expanded = argmax_i64.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, H, L, D)
        K_at_argmax = torch.gather(Ks, dim=2, index=argmax_expanded)  # (B, H, L, D)
        dQ_correction = total_F.unsqueeze(-1) * K_at_argmax  # (B, H, L, D)
        dQ.sub_(dQ_correction)

        # dK[argmax_j,:] -= total_F[j] * Qs[j,:]
        # scatter_add: add -total_F[j] * Qs[j,:] to dK at position argmax_j
        Q_times_totalF = total_F.unsqueeze(-1) * Qs  # (B, H, L, D)
        dK.scatter_add_(dim=2, index=argmax_expanded, src=-Q_times_totalF)

        # ── τ, α gradients ──
        d_tau = torch.zeros(H, device='cuda', dtype=torch.float32)
        d_alpha = torch.zeros(H, device='cuda', dtype=torch.float32)

        for h in range(H):
            q_h = Qs[:, h, :, :]
            k_h = Ks[:, h, :, :]
            v_h = Vf[:, h, :, :]
            do_h = dO[:, h, :, :]
            m_h = m[:, h, :]
            l_h = l[:, h, :]

            s_h = torch.bmm(q_h, k_h.transpose(-2, -1))
            x_stable = s_h - m_h.unsqueeze(-1)
            sigma_val = _sigma_fn_registry(sigma_name, x_stable * alpha[h]).clamp(min=CLAMP_MIN)
            log_q = tau[h] * sigma_val.log().clamp(min=-30)
            q = log_q.exp()
            attn = q / l_h.unsqueeze(-1).clamp(min=CLAMP_MIN)

            dP = torch.bmm(do_h, v_h.transpose(-2, -1))

            log_sigma = sigma_val.log().clamp(min=-30)
            e_log_sigma = (attn * log_sigma).sum(dim=-1, keepdim=True)
            d_attn_dtau = attn * (log_sigma - e_log_sigma)
            d_tau[h] = (dP * d_attn_dtau).sum()

            sigma_p = _sigma_prime_registry(sigma_name)(x_stable * alpha[h]).clamp(min=CLAMP_MIN)
            d_logsigma_dalpha = (sigma_p / sigma_val * x_stable).clamp(-30, 30)
            d_attn_dalpha = attn * tau[h] * (d_logsigma_dalpha - (attn * d_logsigma_dalpha).sum(dim=-1, keepdim=True))
            d_alpha[h] = (dP * d_attn_dalpha).sum()

        d_log_tau = d_tau * torch.sigmoid(log_tau)
        d_log_alpha = d_alpha * torch.exp(log_alpha)

        # Scale correction: dQ/dK from Triton are w.r.t. Qs/Ks = Q*√s, K*√s
        scale_factor = scale ** 0.5
        dQ.mul_(scale_factor)
        dK.mul_(scale_factor)

        return (
            dQ.to(Qs.dtype), dK.to(Ks.dtype), dV.to(Vf.dtype),
            d_log_tau, d_log_alpha, None, None, None,
        )


# ═══════════════════════════════════════════════════════════════════
# Python σ helpers
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("Flash τ-opus kernel test (fixed argmax correction)")
    print("=" * 50)

    # small correctness test
    B, H, M, N, D = 2, 4, 128, 128, 64
    Q = torch.randn(B, H, M, D, device='cuda', requires_grad=True)
    K = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
    V = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
    lt = torch.zeros(H, device='cuda', requires_grad=True)
    la = torch.zeros(H, device='cuda', requires_grad=True)

    out, _ = STauOpusFlashFunction.apply(Q, K, V, lt, la, 'softplus')
    loss = out.sum()
    loss.backward()

    print(f"Forward: {out.shape}")
    print(f"NaN in Q.grad: {torch.isnan(Q.grad).any().item()}")
    print(f"Inf in Q.grad: {torch.isinf(Q.grad).any().item()}")
    print(f"lt_grad: {lt.grad.abs().mean().item():.4f}")
    print(f"la_grad: {la.grad.abs().mean().item():.4f}")

    # Speed test
    print("\nSpeed test (B=4, H=8, L=1500, D=64)...")
    B2, H2, L2, D2 = 4, 8, 1500, 64
    Q2 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
    K2 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
    V2 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
    lt2 = torch.zeros(H2, device='cuda', requires_grad=True)
    la2 = torch.zeros(H2, device='cuda', requires_grad=True)

    for _ in range(3):
        o, _ = STauOpusFlashFunction.apply(Q2, K2, V2, lt2, la2, 'softplus')
        o.sum().backward()
    torch.cuda.synchronize()

    import time
    t0 = time.time()
    for _ in range(10):
        o, _ = STauOpusFlashFunction.apply(Q2, K2, V2, lt2, la2, 'softplus')
        o.sum().backward()
    torch.cuda.synchronize()
    t_flash = (time.time() - t0) / 10
    print(f"Flash kernel: {t_flash*1000:.1f}ms/layer")

    # Compare with old operator (if available)
    try:
        from stau_opus_operator import STauOpusAttentionFunction
        Q2o = Q2.detach().clone().requires_grad_(True)
        K2o = K2.detach().clone().requires_grad_(True)
        V2o = V2.detach().clone().requires_grad_(True)
        lt2o = torch.zeros(H2, device='cuda', requires_grad=True)
        la2o = torch.zeros(H2, device='cuda', requires_grad=True)

        for _ in range(3):
            o, _ = STauOpusAttentionFunction.apply(Q2o, K2o, V2o, lt2o, la2o, 'softplus', None, None)
            o.sum().backward()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(10):
            o, _ = STauOpusAttentionFunction.apply(Q2o, K2o, V2o, lt2o, la2o, 'softplus', None, None)
            o.sum().backward()
        torch.cuda.synchronize()
        t_old = (time.time() - t0) / 10
        print(f"Old operator: {t_old*1000:.1f}ms/layer")
        print(f"Speedup: {t_old/t_flash:.1f}x")
    except ImportError:
        print("Old operator not available for speed comparison.")

    print("\nDone.")
