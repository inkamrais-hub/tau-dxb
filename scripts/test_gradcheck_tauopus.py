"""
τ-opus backward 数值梯度检查。
用 torch.autograd.gradcheck 验证整个 forward/backward 的梯度正确性。

检查内容：
  1. 标准小 case（B=2, H=1, Lq=4, Lk=4, D=8）的完整梯度
  2. 交叉注意力 case（Lq=3, Lk=7）不同 Q/K 长度
  3. 三种 σ 函数（softplus, sigmoid, exp）
  4. 手动计算 argmax correction 项，验证其系数

用法：
  python test_gradcheck_tauopus.py
"""
import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from stau_opus_operator import STauOpusAttentionFunction, _SIGMA_REGISTRY, _SIGMA_PRIME_REGISTRY

CLAMP_MIN = 1e-8
torch.set_printoptions(precision=10, sci_mode=False)


def manual_forward(Q, K, V, tau, alpha, sigma_name, scale=1.0):
    """手动计算 τ-opus forward，返回 output + 中间值用于梯度验证。"""
    B, H, Lq, D = Q.shape
    _, _, Lk, _ = K.shape

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B,H,Lq,Lk)
    s_max = scores.max(dim=-1, keepdim=True).values
    s_stable = scores - s_max  # (B,H,Lq,Lk)

    sigma_fn = _SIGMA_REGISTRY[sigma_name]
    sigma_val = sigma_fn(s_stable * alpha).clamp(min=CLAMP_MIN)
    q = sigma_val.pow(tau).clamp(min=CLAMP_MIN)
    Z = q.sum(dim=-1, keepdim=True).clamp(min=CLAMP_MIN)
    attn = q / Z
    out = torch.matmul(attn, V)
    return out, attn, scores, s_max, s_stable, sigma_val, q, Z


def manual_backward_via_grad(dQ_wrt_out, Q, K, V, tau, alpha, sigma_name, scale=1.0):
    """手动推导 backward（不含 argmax correction），用于对比。"""
    B, H, Lq, D = Q.shape
    _, _, Lk, _ = K.shape

    with torch.enable_grad():
        Qg = Q.detach().requires_grad_(True)
        Kg = K.detach().requires_grad_(True)
        Vg = V.detach().requires_grad_(True)
        out, attn, scores, s_max, s_stable, sigma_val, q, Z = manual_forward(
            Qg, Kg, Vg, tau, alpha, sigma_name, scale)

        out.backward(dQ_wrt_out)

    return Qg.grad, Kg.grad, Vg.grad, attn


def check_argmax_correction(sigma_name="softplus"):
    """单独检查 argmax correction 的数值正确性。

    原理：比较「完整反向」和「去掉 argmax correction 的反向」的差异，
    验证 correction = total_F_i · K[argmax_i] 是否精确匹配。
    """
    torch.manual_seed(42)
    B, H, Lq, Lk, D = 2, 1, 5, 5, 8
    dtype = torch.float64

    Q = torch.randn(B, H, Lq, D, dtype=dtype, device='cuda', requires_grad=True)
    K = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
    V = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)

    log_tau = torch.log(torch.tensor([2.0], device='cuda', dtype=dtype) - 1.0)
    log_alpha = torch.log(torch.tensor([1.5], device='cuda', dtype=dtype))

    tau = (F.softplus(log_tau) + 1.0).view(1, 1, 1, 1)
    alpha = torch.exp(log_alpha).view(1, 1, 1, 1)
    scale = D ** -0.5

    # Forward
    out, attn, scores, s_max, s_stable, sigma_val, q, Z = manual_forward(
        Q, K, V, tau, alpha, sigma_name, scale)

    dO = torch.randn_like(out)
    out.backward(dO)

    # Get full gradients (with argmax correction)
    dQ_full = Q.grad.clone()
    dK_full = K.grad.clone()

    # ── Manually compute F_ij and argmax correction ──
    sigma_fn = _SIGMA_REGISTRY[sigma_name]
    sigma_prime_fn = _SIGMA_PRIME_REGISTRY[sigma_name]

    grad_attn = torch.matmul(dO, V.transpose(-2, -1))
    weighted = (grad_attn * attn).sum(dim=-1, keepdim=True)
    d_log_sigma = tau * attn * (grad_attn - weighted)

    sigma_val_recomp = sigma_fn(s_stable * alpha).clamp(min=CLAMP_MIN)
    sigma_p = sigma_prime_fn(s_stable * alpha).clamp(min=CLAMP_MIN)
    F_ij = d_log_sigma * (sigma_p / sigma_val_recomp) * alpha  # (B,H,Lq,Lk)

    # total_F_i = Σ_j F_ij
    total_F_i = F_ij.sum(dim=-1)  # (B, H, Lq)

    # argmax correction for dQ: dQ_i -= total_F_i · K[argmax_i]
    argmax = scores.argmax(dim=-1, keepdim=True)  # (B,H,Lq,1)
    argmax_expanded = argmax.unsqueeze(-1).expand(-1, -1, -1, -1, D).squeeze(-2)  # (B,H,Lq,D)
    K_at_argmax = torch.gather(K, dim=2, index=argmax_expanded)

    dQ_correction_manual = total_F_i.unsqueeze(-1) * K_at_argmax  # (B,H,Lq,D)

    # Compute dQ without argmax correction: F_ij @ K_j
    dQ_no_correction = torch.matmul(F_ij, K) * scale  # (B,H,Lq,D)

    # dQ with correction = dQ_no_correction - dQ_correction (both terms have *scale)
    dQ_with_correction = dQ_no_correction - dQ_correction_manual * scale

    # Compare with PyTorch autograd
    dQ_diff = (dQ_with_correction - dQ_full).abs().max().item()

    # Verify the correction formula analytically
    # dQ_correction[i,:] = total_F_i · K[argmax_i]
    # Check that this matches the difference between no-correction and full gradient
    correction_from_diff = (dQ_no_correction - dQ_full).abs().max().item()

    print(f"\n  σ={sigma_name}:")
    print(f"    argmax correction formula vs full grad: diff={dQ_diff:.2e}")
    print(f"    correction magnitude: max|correction|={dQ_correction_manual.abs().max().item():.2e}")
    print(f"    total_F range: [{total_F_i.min().item():.2e}, {total_F_i.max().item():.2e}]")
    print(f"    correction from diff: max|dQ_no_correction - dQ_full|={correction_from_diff:.2e}")

    # Verify that correction is non-zero (proof that it IS needed)
    if correction_from_diff < 1e-10:
        print("    ⚠️  correction is effectively zero — argmax correction is unnecessary")
    else:
        print(f"    ✅ correction is non-zero — the formula is meaningful")

    # dK correction: dK[argmax_i,:] -= total_F_i · Q[i,:] * scale
    # scatter_add sums over all i where argmax_i = j
    Q_times_totalF = total_F_i.unsqueeze(-1) * Q * scale  # (B,H,Lq,D)
    dK_no_correction = torch.matmul(F_ij.transpose(-2, -1), Q) * scale  # (B,H,Lk,D)

    dK_correction = torch.zeros_like(dK_no_correction)
    dK_correction.scatter_add_(dim=2, index=argmax_expanded, src=-Q_times_totalF)
    dK_with_correction = dK_no_correction + dK_correction

    dK_diff = (dK_with_correction - dK_full).abs().max().item()
    print(f"    dK argmax correction vs full grad: diff={dK_diff:.2e}")

    return dQ_diff, dK_diff


def gradcheck_all_sigma():
    """使用 torch.autograd.gradcheck 做完整数值梯度检查。
    
    注意：gradcheck 对含有 clamp 和 pow 的 op 可能给出 false positive，
    这里同时跑有限差份作为辅助验证。
    """
    for sigma_name in ["softplus", "sigmoid", "exp"]:
        print(f"\n{'='*60}")
        print(f"gradcheck: σ={sigma_name}")

        # Standard case: Lq=Lk
        torch.manual_seed(42)
        B, H, Lq, Lk, D = 2, 1, 4, 4, 8
        dtype = torch.float64

        Q = torch.randn(B, H, Lq, D, dtype=dtype, device='cuda', requires_grad=True)
        K = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
        V = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
        log_tau = torch.log(torch.tensor([2.5], device='cuda', dtype=dtype) - 1.0)
        log_tau.requires_grad_(True)
        log_alpha = torch.log(torch.tensor([1.2], device='cuda', dtype=dtype))
        log_alpha.requires_grad_(True)

        result = torch.autograd.gradcheck(
            STauOpusAttentionFunction.apply,
            (Q, K, V, log_tau, log_alpha, sigma_name, None, None),
            eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=False, nondet_tol=0.0
        )

        print(f"  gradcheck (Lq=Lk=4): {'PASS' if result else 'FAIL'}")
        if not result:
            # Check specific components
            out, _ = STauOpusAttentionFunction.apply(Q, K, V, log_tau, log_alpha, sigma_name)
            loss = out.sum()
            grads_ad = torch.autograd.grad(loss, [Q, K, V, log_tau, log_alpha], create_graph=False)
            
            # Finite difference on Q (most important)
            eps = 1e-6
            Q_flat = Q.detach().reshape(-1)
            fd_errs = []
            for idx in range(0, min(10, Q_flat.shape[0])):  # check first 10 elements
                Q_plus = Q.detach().clone()
                Q_minus = Q.detach().clone()
                Q_plus.reshape(-1)[idx] += eps
                Q_minus.reshape(-1)[idx] -= eps
                loss_plus, _ = STauOpusAttentionFunction.apply(Q_plus, K, V, log_tau, log_alpha, sigma_name)
                loss_minus, _ = STauOpusAttentionFunction.apply(Q_minus, K, V, log_tau, log_alpha, sigma_name)
                fd_grad = (loss_plus.sum() - loss_minus.sum()) / (2 * eps)
                ad_grad = grads_ad[0].reshape(-1)[idx].item()
                fd_errs.append(abs(ad_grad - fd_grad) / max(abs(fd_grad), 1e-8))
            avg_err = sum(fd_errs) / len(fd_errs)
            print(f"    FD check (Q, first 10): avg_rel_err={avg_err:.2e}")

        # Cross-attention case: Lq ≠ Lk
        B, H, Lq, Lk, D = 2, 1, 3, 7, 8
        Q2 = torch.randn(B, H, Lq, D, dtype=dtype, device='cuda', requires_grad=True)
        K2 = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
        V2 = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)

        result2 = torch.autograd.gradcheck(
            STauOpusAttentionFunction.apply,
            (Q2, K2, V2, log_tau, log_alpha, sigma_name, None, None),
            eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False, nondet_tol=0.0
        )

        print(f"  gradcheck (Lq=3, Lk=7): {'✅ PASS' if result2 else '❌ FAIL'}")

        # Multi-head case
        B, H, Lq, Lk, D = 1, 4, 5, 5, 8
        Q3 = torch.randn(B, H, Lq, D, dtype=dtype, device='cuda', requires_grad=True)
        K3 = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
        V3 = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
        log_tau3 = torch.log(torch.tensor([2.0, 3.0, 1.5, 4.0], device='cuda', dtype=dtype) - 1.0)
        log_tau3.requires_grad_(True)
        log_alpha3 = torch.log(torch.tensor([1.0, 2.0, 0.5, 1.5], device='cuda', dtype=dtype))
        log_alpha3.requires_grad_(True)

        result3 = torch.autograd.gradcheck(
            STauOpusAttentionFunction.apply,
            (Q3, K3, V3, log_tau3, log_alpha3, sigma_name, None, None),
            eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False, nondet_tol=0.0
        )

        print(f"  gradcheck (multi-head H=4): {'✅ PASS' if result3 else '❌ FAIL'}")


def check_alpha_grad_is_nonzero(sigma_name="softplus"):
    """验证 α 的梯度确实非零（如果 α 对 P 无影响，梯度应为 0）。"""
    torch.manual_seed(42)
    B, H, Lq, Lk, D = 2, 1, 4, 4, 8
    dtype = torch.float64

    Q = torch.randn(B, H, Lq, D, dtype=dtype, device='cuda', requires_grad=True)
    K = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
    V = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)

    log_tau = torch.log(torch.tensor([2.5], device='cuda', dtype=dtype) - 1.0)
    log_alpha = torch.log(torch.tensor([1.5], device='cuda', dtype=dtype))
    log_alpha.requires_grad_(True)

    # Forward-backward
    out, _ = STauOpusAttentionFunction.apply(Q, K, V, log_tau, log_alpha, sigma_name)
    dO = torch.randn_like(out)
    out.backward(dO)

    alpha_grad = log_alpha.grad.clone().item()
    print(f"\n  σ={sigma_name}: log_alpha.grad = {alpha_grad:.6e}")

    if abs(alpha_grad) < 1e-10:
        print("    ❌ log_alpha grad is ZERO — α is a dead parameter!")
    else:
        print(f"    ✅ log_alpha grad is non-zero — α has learnable signal")

    return abs(alpha_grad) > 1e-10


def finite_difference_check(sigma_name="softplus"):
    """有限差分验证 α 和 τ 的梯度。"""
    torch.manual_seed(42)
    B, H, Lq, Lk, D = 2, 1, 4, 4, 8
    dtype = torch.float64
    eps = 1e-6

    Q = torch.randn(B, H, Lq, D, dtype=dtype, device='cuda', requires_grad=True)
    K = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)
    V = torch.randn(B, H, Lk, D, dtype=dtype, device='cuda', requires_grad=True)

    log_tau = torch.log(torch.tensor([2.5], device='cuda', dtype=dtype) - 1.0)
    log_alpha = torch.log(torch.tensor([1.5], device='cuda', dtype=dtype))
    log_tau.requires_grad_(True)
    log_alpha.requires_grad_(True)

    # Autograd backward
    out, _ = STauOpusAttentionFunction.apply(Q, K, V, log_tau, log_alpha, sigma_name)
    loss = out.sum()
    loss.backward()
    grad_alpha_ad = log_alpha.grad.item()

    # Finite difference
    loss_plus, _ = STauOpusAttentionFunction.apply(Q, K, V, log_tau, (log_alpha + eps).detach(), sigma_name)
    loss_minus, _ = STauOpusAttentionFunction.apply(Q, K, V, log_tau, (log_alpha - eps).detach(), sigma_name)
    grad_alpha_fd = (loss_plus.sum() - loss_minus.sum()) / (2 * eps)

    # Also check tau
    grad_tau_ad = log_tau.grad.item()
    loss_plus_t, _ = STauOpusAttentionFunction.apply(Q, K, V, (log_tau + eps).detach(), log_alpha, sigma_name)
    loss_minus_t, _ = STauOpusAttentionFunction.apply(Q, K, V, (log_tau - eps).detach(), log_alpha, sigma_name)
    grad_tau_fd = (loss_plus_t.sum() - loss_minus_t.sum()) / (2 * eps)

    err_alpha = abs(grad_alpha_ad - grad_alpha_fd)
    err_tau = abs(grad_tau_ad - grad_tau_fd)

    print(f"\n  σ={sigma_name} 有限差分验证:")
    print(f"    dL/d(log_α) AD={grad_alpha_ad:.6e}  FD={grad_alpha_fd:.6e}  err={err_alpha:.2e}")
    print(f"    dL/d(log_τ) AD={grad_tau_ad:.6e}  FD={grad_tau_fd:.6e}  err={err_tau:.2e}")

    return err_alpha, err_tau


if __name__ == "__main__":
    print("=" * 60)
    print("τ-opus gradient verification")
    print("=" * 60)
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 1. gradcheck
    print(f"\n{'#'*60}")
    print("# 1. torch.autograd.gradcheck (numerical vs analytical)")
    print(f"{'#'*60}")
    gradcheck_all_sigma()

    # 2. Argmax correction analysis
    print(f"\n{'#'*60}")
    print("# 2. Argmax correction analysis")
    print(f"{'#'*60}")
    for sigma in ["softplus", "sigmoid", "exp"]:
        check_argmax_correction(sigma)

    # 3. α gradient check
    print(f"\n{'#'*60}")
    print("# 3. α gradient: is it learnable?")
    print(f"{'#'*60}")
    all_alive = True
    for sigma in ["softplus", "sigmoid", "exp"]:
        alive = check_alpha_grad_is_nonzero(sigma)
        all_alive = all_alive and alive
    if all_alive:
        print(f"\n  ✅ α is a learnable parameter for all σ types")
    else:
        print(f"\n  ⚠️ α gradient is zero for at least one σ type!")

    # 4. Finite difference check
    print(f"\n{'#'*60}")
    print("# 4. Finite difference check (α and τ)")
    print(f"{'#'*60}")
    for sigma in ["softplus", "sigmoid", "exp"]:
        finite_difference_check(sigma)

    print(f"\n{'='*60}")
    print("Done.")
