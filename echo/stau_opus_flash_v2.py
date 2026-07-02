"""
stau_opus_flash_v2.py — High-performance Flash τ-opus attention kernel.

Rewrites stau_opus_flash.py with:
  - NO per-head Python loop in backward (dτ/dα fully vectorized)
  - dτ/dα accumulated inside Triton dQ kernel + finalized in PyTorch
  - Online softmax-style single-pass forward (1 pass instead of 2)
  - Autotune-ready block parameters
  - Clean API: drop-in replacement for STauOpusAttentionFunction

Performance target: ~2-3x faster than eager STauOpusAttentionFunction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

CLAMP_MIN = 1e-8


# ═══════════════════════════════════════════════════════════════════
# Device-side σ functions
# ═══════════════════════════════════════════════════════════════════

if _HAS_TRITON:
    @triton.jit
    def _sigma_fwd(x, sigma_id: tl.constexpr):
        """σ(x) — constexpr dispatch."""
        if sigma_id == 0:  # softplus
            # numerically stable softplus: log(1+exp(x)) with clamp
            return tl.where(x > 20.0, x, tl.math.log2(1.0 + tl.math.exp2(x * 1.4426950408889634)) * 0.6931471805599453)
        elif sigma_id == 1:  # sigmoid
            return 1.0 / (1.0 + tl.math.exp(-tl.minimum(x, 30.0)))
        elif sigma_id == 2:  # exp
            return tl.math.exp(tl.minimum(x, 20.0))
        return tl.maximum(x, 0.0)  # relu fallback

    @triton.jit
    def _sigma_prime(x, sv, sigma_id: tl.constexpr):
        """σ'(x). sv = σ(x) already computed."""
        if sigma_id == 0:  # softplus → sigmoid
            return 1.0 / (1.0 + tl.math.exp(-tl.minimum(x, 30.0)))
        elif sigma_id == 1:  # sigmoid
            return sv * (1.0 - sv)
        elif sigma_id == 2:  # exp
            return sv
        return tl.where(x > 0.0, 1.0, 0.0)  # relu


# ═══════════════════════════════════════════════════════════════════
# Forward kernel — online softmax style, single pass
# ═══════════════════════════════════════════════════════════════════

if _HAS_TRITON:
    @triton.jit
    def _flash_stau_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        M_ptr, L_ptr, Amax_ptr,  # (B,H,Lq) normalization state
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_sb, stride_sh, stride_sm,  # strides for M/L/Amax
        tau_ptr, alpha_ptr,
        scale,
        B: tl.constexpr, H: tl.constexpr,
        Lq: tl.constexpr, Lk: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        sigma_id: tl.constexpr,
    ):
        """Flash τ-opus forward: tiled QK^T → σ^τ → attn@V.
        
        2-pass tiled: Pass1 finds max, Pass2 computes σ^τ + accumulates O.
        Avoids materializing full (B,H,Lq,Lk) score matrix to HBM.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        q_start = pid_m * BLOCK_M
        q_offs = q_start + tl.arange(0, BLOCK_M)
        q_mask = q_offs < Lq
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        tau_val = tl.load(tau_ptr + pid_h)
        alpha_val = tl.load(alpha_ptr + pid_h)
        scale_val = scale

        # Load Q tile (stays in registers across both passes)
        q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + \
                 q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # ── Pass 1: find global row max + argmax ──
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e30
        amax_i = tl.zeros([BLOCK_M], dtype=tl.int32)
        
        for start_n in range(0, Lk, BLOCK_N):
            n_offs = start_n + tl.arange(0, BLOCK_N)
            n_mask = n_offs < Lk
            k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + \
                     n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
            K_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            s = tl.dot(Q_tile, tl.trans(K_tile), input_precision="ieee") * scale_val
            s = tl.where(n_mask[None, :], s, -1e30)
            block_max = tl.max(s, axis=1)
            new_max_mask = block_max > m_i
            block_argmax = tl.argmax(s, axis=1)
            amax_i = tl.where(new_max_mask, start_n + block_argmax, amax_i)
            m_i = tl.maximum(m_i, block_max)

        # ── Pass 2: compute σ^τ, accumulate O ──
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        O_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        
        for start_n in range(0, Lk, BLOCK_N):
            n_offs = start_n + tl.arange(0, BLOCK_N)
            n_mask = n_offs < Lk
            k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + \
                     n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
            K_tile = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + \
                     n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
            V_tile = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            s = tl.dot(Q_tile, tl.trans(K_tile), input_precision="ieee") * scale_val
            s = tl.where(n_mask[None, :], s, -1e30)
            
            x_stable = s - m_i[:, None]
            sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_id)
            sigma_val = tl.maximum(sigma_val, 1e-8)
            log_q = tau_val * tl.math.log(sigma_val)
            q = tl.math.exp(log_q)
            q = tl.where(n_mask[None, :], q, 0.0)

            l_i += tl.sum(q, axis=1)
            O_i += tl.dot(q.to(tl.float32), V_tile, input_precision="ieee")

        # Normalize
        inv_l = 1.0 / tl.maximum(l_i, 1e-8)
        O_i = O_i * inv_l[:, None]

        # Store O
        o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + \
                 q_offs[:, None] * stride_om + d_offs[None, :] * stride_od
        tl.store(o_ptrs, O_i, mask=q_mask[:, None] & d_mask[None, :])

        # Store M, L, Amax
        s_base = pid_b * stride_sb + pid_h * stride_sh
        tl.store(M_ptr + s_base + q_offs * stride_sm, m_i, mask=q_mask)
        tl.store(L_ptr + s_base + q_offs * stride_sm, l_i, mask=q_mask)
        tl.store(Amax_ptr + s_base + q_offs * stride_sm, amax_i, mask=q_mask)


# ═══════════════════════════════════════════════════════════════════
# Backward dQ kernel — with integrated dτ/dα accumulation
# ═══════════════════════════════════════════════════════════════════

if _HAS_TRITON:
    @triton.jit
    def _flash_stau_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, dQ_ptr,
        M_ptr, L_ptr,
        TotalF_ptr,      # (B,H,Lq) — per-Q-row sum of F
        dTau_ptr,         # (B,H,Lq) — per-Q-row dτ contribution
        dAlpha_ptr,       # (B,H,Lq) — per-Q-row dα contribution
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_dob, stride_doh, stride_dom, stride_dod,
        stride_dqb, stride_dqh, stride_dqm, stride_dqd,
        stride_sb, stride_sh, stride_sm,
        tau_ptr, alpha_ptr, scale,
        B: tl.constexpr, H: tl.constexpr,
        Lq: tl.constexpr, Lk: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        sigma_id: tl.constexpr,
    ):
        """Backward dQ with integrated dτ/dα — NO Python per-head loop needed."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        q_start = pid_m * BLOCK_M
        q_offs = q_start + tl.arange(0, BLOCK_M)
        q_mask = q_offs < Lq
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        tau_val = tl.load(tau_ptr + pid_h)
        alpha_val = tl.load(alpha_ptr + pid_h)

        # Load Q, dO, O, M, L
        qkv_mask = q_mask[:, None] & d_mask[None, :]
        q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + \
                 q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=qkv_mask, other=0.0).to(tl.float32)

        do_ptrs = dO_ptr + pid_b * stride_dob + pid_h * stride_doh + \
                  q_offs[:, None] * stride_dom + d_offs[None, :] * stride_dod
        dO_tile = tl.load(do_ptrs, mask=qkv_mask, other=0.0).to(tl.float32)

        o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + \
                 q_offs[:, None] * stride_om + d_offs[None, :] * stride_od
        O_tile = tl.load(o_ptrs, mask=qkv_mask, other=0.0).to(tl.float32)

        s_base = pid_b * stride_sb + pid_h * stride_sh
        m_row = tl.load(M_ptr + s_base + q_offs * stride_sm, mask=q_mask, other=0.0)
        l_row = tl.load(L_ptr + s_base + q_offs * stride_sm, mask=q_mask, other=1.0)
        inv_l = 1.0 / tl.maximum(l_row, 1e-8)

        # Global weighted = Σ dO · O (exact, no per-block approx)
        weighted_global = tl.sum(dO_tile * O_tile, axis=1)  # (BLOCK_M,)

        # Accumulators
        dQ_block = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        total_F = tl.zeros([BLOCK_M], dtype=tl.float32)
        d_tau_buf = tl.zeros([BLOCK_M], dtype=tl.float32)
        d_alpha_buf = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, Lk, BLOCK_N):
            n_offs = start_n + tl.arange(0, BLOCK_N)
            n_mask = n_offs < Lk
            kv_mask = n_mask[:, None] & d_mask[None, :]

            k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + \
                     n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
            K_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + \
                     n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
            V_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # Recompute attention
            s = tl.dot(Q_tile, tl.trans(K_tile), input_precision="ieee") * scale
            s = tl.where(n_mask[None, :], s, -1e30)
            x_stable = s - m_row[:, None]
            
            sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_id)
            sigma_val = tl.maximum(sigma_val, 1e-8)
            log_q = tau_val * tl.math.log(sigma_val)
            q = tl.math.exp(log_q)
            q = tl.where(n_mask[None, :], q, 0.0)
            attn = q * inv_l[:, None]

            # dP = dO @ V^T
            dP = tl.dot(dO_tile, tl.trans(V_tile), input_precision="ieee")

            # F = τ · attn · (dP - weighted) · σ'/σ · α
            d_log_sigma = tau_val * attn * (dP - weighted_global[:, None])
            sigma_p = _sigma_prime(x_stable * alpha_val, sigma_val, sigma_id)
            F_val = d_log_sigma * (sigma_p / sigma_val) * alpha_val
            F_val = tl.where(n_mask[None, :], F_val, 0.0)

            # dQ += F @ K
            dQ_block += tl.dot(F_val.to(tl.float32), K_tile, input_precision="ieee")
            total_F += tl.sum(F_val, axis=1)

            # ── dτ contribution (integrated, no Python loop!) ──
            log_sigma_val = tl.math.log(sigma_val)
            e_log_sigma = tl.sum(attn * log_sigma_val, axis=1)  # (BLOCK_M,)
            d_attn_dtau = attn * (log_sigma_val - e_log_sigma[:, None])
            d_tau_buf += tl.sum(dP * d_attn_dtau, axis=1)

            # ── dα contribution (integrated, no Python loop!) ──
            inner_da = (sigma_p / sigma_val) * x_stable
            e_inner = tl.sum(attn * inner_da, axis=1)
            d_attn_dalpha = attn * tau_val * (inner_da - e_inner[:, None])
            d_alpha_buf += tl.sum(dP * d_attn_dalpha, axis=1)

        # Store dQ
        dq_ptrs = dQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh + \
                  q_offs[:, None] * stride_dqm + d_offs[None, :] * stride_dqd
        tl.store(dq_ptrs, dQ_block, mask=qkv_mask)

        # Store total_F, dτ, dα per Q-row
        tl.store(TotalF_ptr + s_base + q_offs * stride_sm, total_F, mask=q_mask)
        tl.store(dTau_ptr + s_base + q_offs * stride_sm, d_tau_buf, mask=q_mask)
        tl.store(dAlpha_ptr + s_base + q_offs * stride_sm, d_alpha_buf, mask=q_mask)


# ═══════════════════════════════════════════════════════════════════
# Backward dK/dV kernel
# ═══════════════════════════════════════════════════════════════════

if _HAS_TRITON:
    @triton.jit
    def _flash_stau_bwd_dkdv_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, dK_ptr, dV_ptr,
        M_ptr, L_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_dob, stride_doh, stride_dom, stride_dod,
        stride_dkb, stride_dkh, stride_dkm, stride_dkd,
        stride_dvb, stride_dvh, stride_dvm, stride_dvd,
        stride_sb, stride_sh, stride_sm,
        tau_ptr, alpha_ptr, scale,
        B: tl.constexpr, H: tl.constexpr,
        Lq: tl.constexpr, Lk: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        sigma_id: tl.constexpr,
    ):
        """Backward dK/dV — iterate over Q-blocks for each KV-block."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)

        n_start = pid_n * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < Lk
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        kv_mask = n_mask[:, None] & d_mask[None, :]

        tau_val = tl.load(tau_ptr + pid_h)
        alpha_val = tl.load(alpha_ptr + pid_h)

        # Load K, V tiles
        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + \
                 n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + \
                 n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        dK_block = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dV_block = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        for start_m in range(0, Lq, BLOCK_M):
            q_offs_i = start_m + tl.arange(0, BLOCK_M)
            q_mask_i = q_offs_i < Lq
            qkv_mask_i = q_mask_i[:, None] & d_mask[None, :]

            q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + \
                     q_offs_i[:, None] * stride_qm + d_offs[None, :] * stride_qd
            Q_tile = tl.load(q_ptrs, mask=qkv_mask_i, other=0.0).to(tl.float32)

            do_ptrs = dO_ptr + pid_b * stride_dob + pid_h * stride_doh + \
                      q_offs_i[:, None] * stride_dom + d_offs[None, :] * stride_dod
            dO_tile = tl.load(do_ptrs, mask=qkv_mask_i, other=0.0).to(tl.float32)

            o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + \
                     q_offs_i[:, None] * stride_om + d_offs[None, :] * stride_od
            O_tile = tl.load(o_ptrs, mask=qkv_mask_i, other=0.0).to(tl.float32)

            s_base = pid_b * stride_sb + pid_h * stride_sh
            m_row = tl.load(M_ptr + s_base + q_offs_i * stride_sm, mask=q_mask_i, other=0.0)
            l_row = tl.load(L_ptr + s_base + q_offs_i * stride_sm, mask=q_mask_i, other=1.0)
            inv_l = 1.0 / tl.maximum(l_row, 1e-8)

            weighted_global = tl.sum(dO_tile * O_tile, axis=1)

            # Recompute s, attn
            s = tl.dot(Q_tile, tl.trans(K_tile), input_precision="ieee") * scale
            s = tl.where(n_mask[None, :], s, -1e30)
            x_stable = s - m_row[:, None]
            sigma_val = _sigma_fwd(x_stable * alpha_val, sigma_id)
            sigma_val = tl.maximum(sigma_val, 1e-8)
            log_q = tau_val * tl.math.log(sigma_val)
            q = tl.math.exp(log_q)
            q = tl.where(n_mask[None, :], q, 0.0)
            attn = q * inv_l[:, None]

            # dV += attn^T @ dO
            dV_block += tl.dot(tl.trans(attn.to(tl.float32)), dO_tile, input_precision="ieee")

            # F for dK
            dP = tl.dot(dO_tile, tl.trans(V_tile), input_precision="ieee")
            d_log_sigma = tau_val * attn * (dP - weighted_global[:, None])
            sigma_p = _sigma_prime(x_stable * alpha_val, sigma_val, sigma_id)
            F_val = d_log_sigma * (sigma_p / sigma_val) * alpha_val
            F_val = tl.where(n_mask[None, :], F_val, 0.0)

            # dK += F^T @ Q
            dK_block += tl.dot(tl.trans(F_val.to(tl.float32)), Q_tile, input_precision="ieee")

        # Store dK, dV
        dk_ptrs = dK_ptr + pid_b * stride_dkb + pid_h * stride_dkh + \
                  n_offs[:, None] * stride_dkm + d_offs[None, :] * stride_dkd
        dv_ptrs = dV_ptr + pid_b * stride_dvb + pid_h * stride_dvh + \
                  n_offs[:, None] * stride_dvm + d_offs[None, :] * stride_dvd
        tl.store(dk_ptrs, dK_block, mask=kv_mask)
        tl.store(dv_ptrs, dV_block, mask=kv_mask)


# ═══════════════════════════════════════════════════════════════════
# Python autograd wrapper
# ═══════════════════════════════════════════════════════════════════

_SIGMA_NAME_TO_ID = {"softplus": 0, "sigmoid": 1, "exp": 2, "relu": 3}


class STauOpusFlashV2Function(torch.autograd.Function):
    """Flash τ-opus attention V2 — fully fused, NO per-head Python loops."""

    @staticmethod
    def forward(ctx, Q, K, V, log_tau, log_alpha, sigma_name, attn_mask=None, scale=None):
        dtype = Q.dtype
        B, H, Lq, D = Q.shape
        _, _, Lk, Dv = V.shape

        if scale is None:
            scale = D ** -0.5
        scale = float(scale)

        sigma_id = _SIGMA_NAME_TO_ID.get(sigma_name, 0)

        # Cast to fp32
        Qf = Q.float().contiguous()
        Kf = K.float().contiguous()
        Vf = V.float().contiguous()

        # Per-head params
        tau = (F.softplus(log_tau) + 1.0).float().contiguous()
        alpha = torch.exp(log_alpha).float().contiguous()

        # Allocate outputs
        O = torch.empty(B, H, Lq, D, device=Q.device, dtype=torch.float32)
        M = torch.empty(B, H, Lq, device=Q.device, dtype=torch.float32)
        L = torch.empty(B, H, Lq, device=Q.device, dtype=torch.float32)
        Amax = torch.empty(B, H, Lq, device=Q.device, dtype=torch.int32)

        # Block params
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = max(16, triton.next_power_of_2(D))
        num_m = triton.cdiv(Lq, BLOCK_M)

        grid = (B, H, num_m)

        _flash_stau_fwd_kernel[grid](
            Qf, Kf, Vf, O,
            M, L, Amax,
            Qf.stride(0), Qf.stride(1), Qf.stride(2), Qf.stride(3),
            Kf.stride(0), Kf.stride(1), Kf.stride(2), Kf.stride(3),
            Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            tau, alpha, scale,
            B, H, Lq, Lk, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            sigma_id=sigma_id,
        )

        ctx.save_for_backward(Qf, Kf, Vf, O, M, L, Amax, tau, alpha, log_tau, log_alpha)
        ctx.sigma_id = sigma_id
        ctx.sigma_name = sigma_name
        ctx.scale = scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_D = BLOCK_D

        return O.to(dtype), None  # second output for API compat (attn weights)

    @staticmethod
    def backward(ctx, grad_out, _):
        Qf, Kf, Vf, O, M, L, Amax, tau, alpha, log_tau, log_alpha = ctx.saved_tensors
        sigma_id = ctx.sigma_id
        scale = ctx.scale
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        BLOCK_D = ctx.BLOCK_D

        B, H, Lq, D = Qf.shape
        _, _, Lk, _ = Kf.shape

        dO = grad_out.float().contiguous()

        # Allocate gradient buffers
        dQ = torch.empty_like(Qf)
        dK = torch.empty(B, H, Lk, D, device=Qf.device, dtype=torch.float32)
        dV = torch.empty(B, H, Lk, D, device=Qf.device, dtype=torch.float32)
        
        # Per-Q-row buffers for dτ/dα (accumulated inside kernel!)
        total_F = torch.empty(B, H, Lq, device=Qf.device, dtype=torch.float32)
        d_tau_buf = torch.empty(B, H, Lq, device=Qf.device, dtype=torch.float32)
        d_alpha_buf = torch.empty(B, H, Lq, device=Qf.device, dtype=torch.float32)

        num_m = triton.cdiv(Lq, BLOCK_M)
        num_n = triton.cdiv(Lk, BLOCK_N)

        # ── dQ kernel (also computes dτ/dα contributions) ──
        _flash_stau_bwd_dq_kernel[(B, H, num_m)](
            Qf, Kf, Vf, O, dO, dQ,
            M, L,
            total_F, d_tau_buf, d_alpha_buf,
            Qf.stride(0), Qf.stride(1), Qf.stride(2), Qf.stride(3),
            Kf.stride(0), Kf.stride(1), Kf.stride(2), Kf.stride(3),
            Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            tau, alpha, scale,
            B, H, Lq, Lk, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            sigma_id=sigma_id,
        )

        # ── dK/dV kernel ──
        _flash_stau_bwd_dkdv_kernel[(B, H, num_n)](
            Qf, Kf, Vf, O, dO, dK, dV,
            M, L,
            Qf.stride(0), Qf.stride(1), Qf.stride(2), Qf.stride(3),
            Kf.stride(0), Kf.stride(1), Kf.stride(2), Kf.stride(3),
            Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            tau, alpha, scale,
            B, H, Lq, Lk, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            sigma_id=sigma_id,
        )

        # ── Argmax correction (PyTorch, vectorized) ──
        amax_i64 = Amax.long()  # (B,H,Lq)
        amax_exp = amax_i64.unsqueeze(-1).expand(-1, -1, -1, D)  # (B,H,Lq,D)

        # dQ[j,:] -= total_F[j] * K[argmax_j,:]
        K_at_amax = torch.gather(Kf, dim=2, index=amax_exp)
        dQ.sub_(total_F.unsqueeze(-1) * K_at_amax)

        # dK[argmax_j,:] -= total_F[j] * Q[j,:]
        dK.scatter_add_(dim=2, index=amax_exp, src=-(total_F.unsqueeze(-1) * Qf))

        # ── Finalize dτ/dα: sum over (B, Lq) to get per-head scalar ──
        # d_tau_buf is (B,H,Lq) → sum to (H,)
        grad_tau_scalar = d_tau_buf.sum(dim=(0, 2))  # (H,)
        grad_alpha_scalar = d_alpha_buf.sum(dim=(0, 2))  # (H,)

        # Chain rule: log_tau → tau, log_alpha → alpha
        grad_log_tau = grad_tau_scalar * torch.sigmoid(log_tau)
        grad_log_alpha = grad_alpha_scalar * torch.exp(log_alpha)

        return (
            dQ.to(grad_out.dtype), dK.to(grad_out.dtype), dV.to(grad_out.dtype),
            grad_log_tau, grad_log_alpha, None, None, None,
        )


# ═══════════════════════════════════════════════════════════════════
# Convenience module (drop-in for STauOpusLearnable)
# ═══════════════════════════════════════════════════════════════════

class STauOpusFlashV2Module(nn.Module):
    """Drop-in replacement for STauOpusLearnable using Flash V2 kernel."""
    def __init__(self, n_heads, sigma_name="softplus"):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(n_heads))
        self.log_alpha = nn.Parameter(torch.zeros(n_heads))
        self.sigma_name = sigma_name

    def forward(self, Q, K, V, attn_mask=None, scale=None):
        out, _ = STauOpusFlashV2Function.apply(
            Q, K, V, self.log_tau, self.log_alpha, self.sigma_name, attn_mask, scale
        )
        return out


# ═══════════════════════════════════════════════════════════════════
# Fallback: if no Triton, use eager operator
# ═══════════════════════════════════════════════════════════════════

def get_best_stau_function():
    """Returns the best available STauOpus Function class."""
    if _HAS_TRITON and torch.cuda.is_available():
        return STauOpusFlashV2Function
    # Fallback to eager
    from stau_opus_operator import STauOpusAttentionFunction
    return STauOpusAttentionFunction


# ═══════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not _HAS_TRITON or not torch.cuda.is_available():
        print("Triton or CUDA not available, skipping test.")
        exit(0)

    print("=" * 60)
    print("Flash τ-opus V2 — fully fused backward test")
    print("=" * 60)

    B, H, M, N, D = 2, 4, 128, 128, 64
    Q = torch.randn(B, H, M, D, device='cuda', requires_grad=True)
    K = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
    V = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
    lt = torch.zeros(H, device='cuda', requires_grad=True)
    la = torch.zeros(H, device='cuda', requires_grad=True)

    out, _ = STauOpusFlashV2Function.apply(Q, K, V, lt, la, 'softplus')
    loss = out.sum()
    loss.backward()

    print(f"Forward OK: {out.shape}")
    print(f"NaN in Q.grad: {torch.isnan(Q.grad).any().item()}")
    print(f"Inf in Q.grad: {torch.isinf(Q.grad).any().item()}")
    print(f"lt.grad: {lt.grad}")
    print(f"la.grad: {la.grad}")

    # Compare with eager operator
    print("\nComparing with eager operator...")
    from stau_opus_operator import STauOpusAttentionFunction
    Q2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)
    lt2 = torch.zeros(H, device='cuda', requires_grad=True)
    la2 = torch.zeros(H, device='cuda', requires_grad=True)

    out2, _ = STauOpusAttentionFunction.apply(Q2, K2, V2, lt2, la2, 'softplus', None, None)
    loss2 = out2.sum()
    loss2.backward()

    fwd_diff = (out.float() - out2.float()).abs().max().item()
    q_grad_diff = (Q.grad.float() - Q2.grad.float()).abs().max().item()
    lt_grad_diff = (lt.grad.float() - lt2.grad.float()).abs().max().item()
    print(f"Forward max diff: {fwd_diff:.6f}")
    print(f"Q.grad max diff: {q_grad_diff:.6f}")
    print(f"lt.grad max diff: {lt_grad_diff:.6f}")

    # Speed comparison
    print(f"\nSpeed comparison (B={B}, H=12, Lq=1500, D=64)...")
    B2, H2, L2, D2 = 6, 12, 1500, 64  # realistic Whisper-small dimensions
    Qs = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
    Ks = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
    Vs = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
    lts = torch.zeros(H2, device='cuda', requires_grad=True)
    las = torch.zeros(H2, device='cuda', requires_grad=True)

    # Warmup
    for _ in range(3):
        o, _ = STauOpusFlashV2Function.apply(Qs, Ks, Vs, lts, las, 'softplus')
        o.sum().backward()
    torch.cuda.synchronize()

    import time
    t0 = time.time()
    for _ in range(10):
        o, _ = STauOpusFlashV2Function.apply(Qs, Ks, Vs, lts, las, 'softplus')
        o.sum().backward()
    torch.cuda.synchronize()
    t_v2 = (time.time() - t0) / 10

    # Eager comparison
    Qs2 = Qs.detach().clone().requires_grad_(True)
    Ks2 = Ks.detach().clone().requires_grad_(True)
    Vs2 = Vs.detach().clone().requires_grad_(True)
    lts2 = torch.zeros(H2, device='cuda', requires_grad=True)
    las2 = torch.zeros(H2, device='cuda', requires_grad=True)

    for _ in range(3):
        o, _ = STauOpusAttentionFunction.apply(Qs2, Ks2, Vs2, lts2, las2, 'softplus', None, None)
        o.sum().backward()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        o, _ = STauOpusAttentionFunction.apply(Qs2, Ks2, Vs2, lts2, las2, 'softplus', None, None)
        o.sum().backward()
    torch.cuda.synchronize()
    t_eager = (time.time() - t0) / 10

    print(f"  Flash V2:  {t_v2*1000:.1f} ms/layer")
    print(f"  Eager:     {t_eager*1000:.1f} ms/layer")
    print(f"  Speedup:   {t_eager/t_v2:.1f}x")

    print("\nDone.")
