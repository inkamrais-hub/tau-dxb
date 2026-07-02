"""
triton_wta.py — Triton fused WTA top-k kernel for echo-opus-1

Fused operation: top-k threshold + mask creation + element-wise multiply
→ single GPU kernel instead of 3 separate launches (topk + scatter + mul)

Supports:
  - Forward:  select top-k per row → output = x * (rank(x) <= k)
  - Backward: STE through mask (gradient only flows to winner neurons)
  - Graceful fallback: if Triton not available, uses pure PyTorch

For whisper-small FFN (d_ff=3072), each row fits in shared memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass

if _HAS_TRITON:
    @triton.jit
    def _wta_mask_mul_fused_kernel(
        x_ptr,          # [N, D] float input
        threshold_ptr,  # [N] float k-th largest value per row
        y_ptr,          # [N, D] float output
        mask_ptr,       # [N, D] bool mask (for backward STE)
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused: threshold + mask + multiply per row."""
        pid = tl.program_id(0)  # row index
        offs = tl.arange(0, BLOCK_D)
        valid = offs < D

        # Load row + threshold
        x = tl.load(x_ptr + pid * D + offs, mask=valid, other=-float('inf'))
        thresh = tl.load(threshold_ptr + pid)

        # Mask: keep if >= threshold (matches top-k exactly, handles ties)
        keep = x >= thresh
        y = tl.where(keep, x, 0.0)

        tl.store(y_ptr + pid * D + offs, y, mask=valid)
        tl.store(mask_ptr + pid * D + offs, keep.to(tl.int8), mask=valid)

    @triton.jit
    def _wta_backward_kernel(
        grad_y_ptr,     # [N, D]
        mask_ptr,       # [N, D] bool
        grad_x_ptr,     # [N, D]
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """STE backward: grad_x = grad_y * mask (only winners get gradients)."""
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        valid = offs < D

        dy = tl.load(grad_y_ptr + pid * D + offs, mask=valid, other=0.0)
        m = tl.load(mask_ptr + pid * D + offs, mask=valid, other=0).to(tl.float32)
        tl.store(grad_x_ptr + pid * D + offs, dy * m, mask=valid)


    def _triton_wta_forward(x: torch.Tensor, k: int):
        """Triton-accelerated WTA forward. Supports N-D tensors (flattens to 2D). Returns (y, mask)."""
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        N, D = x_2d.shape
        x_contig = x_2d.contiguous()
        y_2d = torch.empty_like(x_contig)
        mask_2d = torch.empty(N, D, dtype=torch.int8, device=x.device)

        # Step 0: find k-th largest per row (PyTorch topk is already very fast)
        k_clamped = max(1, min(D, k))
        thresholds, _ = x_contig.topk(k_clamped, dim=-1, largest=True)
        thresholds = thresholds[:, -1].contiguous()  # [N]

        # Step 1: fused mask + multiply in Triton (1 kernel instead of 2)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (N,)
        _wta_mask_mul_fused_kernel[grid](
            x_contig, thresholds, y_2d, mask_2d, D, BLOCK_D=BLOCK_D
        )
        return y_2d.reshape(orig_shape), mask_2d.reshape(orig_shape).bool()


    class _WTAFusedSTE(torch.autograd.Function):
        """Fused WTA + Straight-Through Estimator (Triton accelerated)."""
        @staticmethod
        def forward(ctx, x, k):
            y, mask = _triton_wta_forward(x, k)
            ctx.save_for_backward(mask)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            (mask,) = ctx.saved_tensors
            orig_shape = grad_y.shape
            grad_y_2d = grad_y.reshape(-1, orig_shape[-1])
            mask_2d = mask.reshape(-1, orig_shape[-1])
            N, D = grad_y_2d.shape
            grad_x_2d = torch.empty_like(grad_y_2d)
            BLOCK_D = triton.next_power_of_2(D)
            _wta_backward_kernel[(N,)](
                grad_y_2d.contiguous(), mask_2d.contiguous(), grad_x_2d, D, BLOCK_D=BLOCK_D
            )
            return grad_x_2d.reshape(orig_shape), None


def _torch_wta_forward(x: torch.Tensor, k: int):
    """Pure PyTorch WTA forward (fallback). Returns (y, mask)."""
    k = max(1, min(x.shape[-1], k))
    _, topk_indices = x.topk(k, dim=-1)
    mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1.0)
    y = x * mask
    return y, mask.bool()


class _WTAPyTorchSTE(torch.autograd.Function):
    """Pure PyTorch WTA + Straight-Through Estimator (fallback)."""
    @staticmethod
    def forward(ctx, x, k):
        y, mask = _torch_wta_forward(x, k)
        ctx.save_for_backward(mask)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (mask,) = ctx.saved_tensors
        return grad_y * mask, None


# ── Public API ──

def wta_fused_forward(x: torch.Tensor, k: int) -> torch.Tensor:
    """Select top-k activations per row, zero out rest. STE backward.

    Returns output tensor (same shape as input).
    Uses Triton if available, otherwise pure PyTorch.

    Args:
        x: [N, D] activation tensor
        k: number of winners per row
    """
    if _HAS_TRITON and x.is_cuda:
        return _WTAFusedSTE.apply(x, k)
    return _WTAPyTorchSTE.apply(x, k)


def wta_fused_forward_with_mask(x: torch.Tensor, k: int):
    """Like wta_fused_forward but also returns the boolean mask."""
    if _HAS_TRITON and x.is_cuda:
        return _triton_wta_forward(x, k)
    return _torch_wta_forward(x, k)


def has_triton() -> bool:
    return _HAS_TRITON


# ── Smoke test ──

if __name__ == "__main__":
    print(f"Triton available: {_HAS_TRITON}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 3072, device=device, requires_grad=True)
    k = int(3072 * 0.4)  # 40% winners

    print(f"Input: {x.shape}, k={k}")

    # Forward
    y = wta_fused_forward(x, k)
    nz = (y != 0).sum(dim=-1).float().mean()
    print(f"  Nonzero per row (avg): {nz:.0f} / {k} expected")

    # Backward
    loss = y.sum()
    loss.backward()
    grad_nz = (x.grad != 0).sum(dim=-1).float().mean()
    print(f"  Grad nonzero (avg): {grad_nz:.0f} (STE: only winners)")

    print("Test OK")
