"""
Isolate multi-block accumulation issue.
Vary: D=64, L=64, 2 KV blocks, Q-block 0 only.
"""
import sys
sys.path.insert(0, 'scripts')
import torch
import triton
import triton.language as tl
from stau_opus_flash import _sigma_fwd, _sigma_prime_fn, _stau_opus_flash_fwd

torch.manual_seed(42)

D = 64
L = 64
BLOCK_M = 32
BLOCK_N = 32

Q = torch.randn(1, 1, L, D, device='cuda')
K = torch.randn(1, 1, L, D, device='cuda')
V = torch.randn(1, 1, L, D, device='cuda')
dO = torch.randn(1, 1, L, D, device='cuda')

scale = float(D ** -0.5)
Qs = Q.float() * (scale ** 0.5)
Ks = K.float() * (scale ** 0.5)
Vf = V.float()

tau_val = 1.0
alpha_val = 1.0

# ── Reference (PyTorch) for Q-block 0 ──
# Full scores + m + l
scores_full = Qs[0,0] @ Ks[0,0].T
m_full = scores_full.max(dim=-1, keepdim=True).values
x_full = scores_full - m_full
sigma_full = torch.nn.functional.softplus(x_full * alpha_val).clamp(min=1e-8)
log_q_full = tau_val * sigma_full.log().clamp(min=-30)
q_full = log_q_full.exp()
l_full = q_full.sum(dim=-1, keepdim=True).clamp(min=1e-8)
attn_full = q_full / l_full
dP_full = dO[0,0] @ Vf[0,0].T
weighted_full = (attn_full * dP_full).sum(dim=-1, keepdim=True)
d_log_sigma_full = tau_val * attn_full * (dP_full - weighted_full)
sigma_p_full = torch.sigmoid(x_full * alpha_val)
F_full = d_log_sigma_full * (sigma_p_full / sigma_full) * alpha_val

# dQ for Q-block 0 only (queries 0..31)
dQ_ref_manual = F_full[:32, :] @ Ks[0,0]  # (32, 64)
total_F_ref_manual = F_full[:32, :].sum(dim=-1)  # (32,)

print(f"Reference F[0,:8]: {[f'{v:.4f}' for v in F_full[0,:8].tolist()]}")

# ── Forward to get m/l ──
m_buf = torch.empty(1, 1, L, device='cuda', dtype=torch.float32)
l_buf = torch.empty(1, 1, L, device='cuda', dtype=torch.float32)
argmax_buf = torch.empty(1, 1, L, device='cuda', dtype=torch.int32)
O_buf = torch.empty_like(Qs)

_stau_opus_flash_fwd[(1,1,2)](
    Qs, Ks, Vf, O_buf, m_buf, l_buf, argmax_buf,
    Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
    Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
    Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
    O_buf.stride(0), O_buf.stride(1), O_buf.stride(2), O_buf.stride(3),
    m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
    l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
    argmax_buf.stride(0), argmax_buf.stride(1), argmax_buf.stride(2),
    torch.tensor([tau_val], device='cuda'),
    torch.tensor([alpha_val], device='cuda'),
    1, 1, L, D,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D, sigma_name='softplus',
)

# Check m/l consistency
print(f"\n--- m/l consistency ---")
print(f"  m_buf[0,0,:8]: {m_buf[0,0,:8].tolist()}")
print(f"  m_ref[:8]:     {m_full[:8,0].tolist()}")
print(f"  m diff: {(m_buf[0,0,:] - m_full[:,0]).abs().max().item():.2e}")
print(f"  l diff: {(l_buf[0,0,:] - l_full[:,0]).abs().max().item():.2e}")

# ── Triton dQ kernel (multi-block, Q-block 0) ──
dQ_triton = torch.empty(1, 1, L, D, device='cuda', dtype=torch.float32)
total_F = torch.empty(1, 1, L, device='cuda', dtype=torch.float32)

from stau_opus_flash import _stau_opus_flash_bwd_dq
grid_dq = (1, 1, 2)  # 2 Q-blocks
_stau_opus_flash_bwd_dq[grid_dq](
    Qs, Ks, Vf, dO.float(), O_buf, dQ_triton,
    m_buf, l_buf, total_F,
    Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
    Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
    Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
    dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
    O_buf.stride(0), O_buf.stride(1), O_buf.stride(2), O_buf.stride(3),
    dQ_triton.stride(0), dQ_triton.stride(1), dQ_triton.stride(2), dQ_triton.stride(3),
    m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
    l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
    total_F.stride(0), total_F.stride(1), total_F.stride(2),
    torch.tensor([tau_val], device='cuda'),
    torch.tensor([alpha_val], device='cuda'),
    1, 1, L, D,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D, sigma_name='softplus',
)

# Compare Q-block 0
dQ_q0_triton = dQ_triton[0,0,:32,:]
total_F_q0_triton = total_F[0,0,:32]

print(f"\n--- Q-block 0: dQ (Σ Fi@Ki), NO argmax correction ---")
print(f"  dQ manual range: [{dQ_ref_manual.min():.4f}, {dQ_ref_manual.max():.4f}]")
print(f"  dQ triton range: [{dQ_q0_triton.min():.4f}, {dQ_q0_triton.max():.4f}]")
print(f"  dQ diff: {(dQ_q0_triton - dQ_ref_manual).abs().max().item():.2e}")
print(f"  total_F manual range: [{total_F_ref_manual.min():.4f}, {total_F_ref_manual.max():.4f}]")
print(f"  total_F triton range: [{total_F_q0_triton.min():.4f}, {total_F_q0_triton.max():.4f}]")
print(f"  total_F diff: {(total_F_q0_triton - total_F_ref_manual).abs().max().item():.2e}")

# Check per-row F diff
print(f"\n--- F per-row check (Q-block 0, first row) ---")
print(f"  F_ref row 0: {[f'{v:.4f}' for v in F_full[0,:8].tolist()]}")

# Also check: what if we use the forward's m_buf and l_buf to recompute manually?
# This isolates m/l mismatch from actual computation difference
m_fwd = m_buf[0,0,:32]  # forward's m for Q-block 0
l_fwd = l_buf[0,0,:32]
inv_l_fwd = 1.0 / l_fwd.clamp(min=1e-8)

k0 = Ks[0,0,:32,:]  # block 0
k1 = Ks[0,0,32:,:]  # block 1
v0 = Vf[0,0,:32,:]
v1 = Vf[0,0,32:,:]
do = dO[0,0,:32,:]

# Manual recompute with FORWARD's m/l (to match kernel exactly)
s_correct = Qs[0,0,:32,:] @ Ks[0,0].T  # full scores for Q-block 0
x_correct = s_correct - m_fwd.unsqueeze(-1)
sigma_correct = torch.nn.functional.softplus(x_correct * alpha_val).clamp(min=1e-8)
log_q_correct = tau_val * sigma_correct.log().clamp(min=-30)
q_correct = log_q_correct.exp()
attn_correct = q_correct / l_fwd.unsqueeze(-1).clamp(min=1e-8)
dP_correct = do @ Vf[0,0].T
weighted_correct = (attn_correct * dP_correct).sum(dim=-1, keepdim=True)
d_log_sigma_correct = tau_val * attn_correct * (dP_correct - weighted_correct)
sigma_p_correct = torch.sigmoid(x_correct * alpha_val)
F_correct = d_log_sigma_correct * (sigma_p_correct / sigma_correct) * alpha_val

dQ_correct = F_correct @ Ks[0,0]
total_F_correct = F_correct.sum(dim=-1)

print(f"\n--- Using forward's m/l ---")
print(f"  dQ diff: {(dQ_q0_triton - dQ_correct).abs().max().item():.2e}")
print(f"  total_F diff: {(total_F_q0_triton - total_F_correct).abs().max().item():.2e}")
print(f"  F ref row 0: {[f'{v:.4f}' for v in F_correct[0,:8].tolist()]}")

# ── Use forward's O for weighted, exactly matching kernel ──
O_fwd = O_buf[0,0,:32,:]  # forward output (normalized)
weighted_from_O = (do * O_fwd).sum(dim=-1, keepdim=True)  # dO·O

# Compute F using kernel's exact weighted
F_from_O = torch.zeros_like(F_correct)
for b in range(2):  # 2 KV blocks
    k_block = Ks[0,0,b*32:(b+1)*32,:]
    v_block = Vf[0,0,b*32:(b+1)*32,:]
    s_block = Qs[0,0,:32,:] @ k_block.T
    x_block = s_block - m_fwd.unsqueeze(-1)
    sigma_block = torch.nn.functional.softplus(x_block * alpha_val).clamp(min=1e-8)
    log_q_block = tau_val * sigma_block.log().clamp(min=-30)
    q_block = log_q_block.exp()
    attn_block = q_block / l_fwd.unsqueeze(-1).clamp(min=1e-8)
    dP_block = do @ v_block.T
    dls_block = tau_val * attn_block * (dP_block - weighted_from_O)  # GLOBAL weighted!
    sp_block = torch.sigmoid(x_block * alpha_val)
    F_from_O[:, b*32:(b+1)*32] = dls_block * (sp_block / sigma_block) * alpha_val

dQ_from_O = F_from_O @ Ks[0,0]
total_F_from_O = F_from_O.sum(dim=-1)

print(f"\n--- Using forward's O for weighted (exact kernel match) ---")
print(f"  dQ from O diff: {(dQ_q0_triton - dQ_from_O).abs().max().item():.2e}")
print(f"  total_F from O diff: {(total_F_q0_triton - total_F_from_O).abs().max().item():.2e}")
print(f"  F_from_O[0,:8]: {[f'{v:.4f}' for v in F_from_O[0,:8].tolist()]}")
print(f"  O_fwd[0,:8]: {[f'{v:.4f}' for v in O_fwd[0,:8].tolist()]}")
print(f"  weighted_from_O[0,0]: {weighted_from_O[0,0].item():.6f}")
print(f"  weighted_correct[0,0]: {weighted_correct[0,0].item():.6f}")

# ── Cross-check: does dO·O = Σ P·(dO·V)? ──
weighted_via_attn = (attn_correct * dP_correct).sum(dim=-1, keepdim=True)
print(f"\n--- Weighted cross-check ---")
print(f"  dO·O vs Σ P·(dO·V) diff: {(weighted_from_O - weighted_via_attn).abs().max().item():.2e}")

# Now check block-by-block with GLOBAL weighted (like kernel)
s_b0 = Qs[0,0,:32,:] @ Ks[0,0,:32,:].T  # (32, 32)
x_b0 = s_b0 - m_fwd.unsqueeze(-1)
sigma_b0 = torch.nn.functional.softplus(x_b0 * alpha_val).clamp(min=1e-8)
log_q_b0 = tau_val * sigma_b0.log().clamp(min=-30)
q_b0 = log_q_b0.exp()
attn_b0 = q_b0 / l_fwd.unsqueeze(-1).clamp(min=1e-8)
dP_b0 = do @ Vf[0,0,:32,:].T
dls_b0 = tau_val * attn_b0 * (dP_b0 - weighted_from_O)  # GLOBAL weighted
sp_b0 = torch.sigmoid(x_b0 * alpha_val)
F_b0 = dls_b0 * (sp_b0 / sigma_b0) * alpha_val

s_b1 = Qs[0,0,:32,:] @ Ks[0,0,32:,:].T  # (32, 32)
x_b1 = s_b1 - m_fwd.unsqueeze(-1)
sigma_b1 = torch.nn.functional.softplus(x_b1 * alpha_val).clamp(min=1e-8)
log_q_b1 = tau_val * sigma_b1.log().clamp(min=-30)
q_b1 = log_q_b1.exp()
attn_b1 = q_b1 / l_fwd.unsqueeze(-1).clamp(min=1e-8)
dP_b1 = do @ Vf[0,0,32:,:].T
dls_b1 = tau_val * attn_b1 * (dP_b1 - weighted_from_O)  # GLOBAL weighted
sp_b1 = torch.sigmoid(x_b1 * alpha_val)
F_b1 = dls_b1 * (sp_b1 / sigma_b1) * alpha_val

# Total by block
dQ_b0_manual = F_b0 @ Ks[0,0,:32,:]  # F_b0 @ K0
dQ_b1_manual = F_b1 @ Ks[0,0,32:,:]  # F_b1 @ K1
dQ_block_manual = dQ_b0_manual + dQ_b1_manual
total_F_block_manual = F_b0.sum(dim=-1) + F_b1.sum(dim=-1)

print(f"\n--- Block-by-block ---")
print(f"  dQ block manual vs triton: diff={(dQ_block_manual - dQ_q0_triton).abs().max().item():.2e}")
print(f"  dQ block0 F@K0 vs triton partial: check per-block")
print(f"  F_b0[0,:8]: {[f'{v:.4f}' for v in F_b0[0,:8].tolist()]}")
print(f"  F_b1[0,:8]: {[f'{v:.4f}' for v in F_b1[0,:8].tolist()]}")

# Most critical: check F_b0 row 0 vs first 32 positions of F_correct row 0
b0_diff = (F_b0[:,:] - F_correct[:,:32]).abs().max().item()
b1_diff = (F_b1[:,:] - F_correct[:,32:]).abs().max().item()
print(f"  F_b0 vs F_correct[:,:32] diff: {b0_diff:.2e}")
print(f"  F_b1 vs F_correct[:,32:] diff: {b1_diff:.2e}")

# ── DIRECT dQ comparison: block-by-block PyTorch vs single matmul vs kernel ──
dQ_b0_py = F_b0 @ Ks[0,0,:32,:]  # block 0: (32,32) @ (32,64)
dQ_b1_py = F_b1 @ Ks[0,0,32:,:]  # block 1: (32,32) @ (32,64)
dQ_py_blocked = dQ_b0_py + dQ_b1_py

print(f"\n--- Direct dQ accumulation comparison ---")
print(f"  dQ PyTorch block-by-block vs single matmul: {(dQ_py_blocked - dQ_from_O).abs().max().item():.2e}")
print(f"  dQ kernel vs PyTorch block-by-block: {(dQ_q0_triton - dQ_py_blocked).abs().max().item():.2e}")
print(f"  dQ kernel vs PyTorch single matmul: {(dQ_q0_triton - dQ_from_O).abs().max().item():.2e}")

# ── Check per-block dQ separately ──
print(f"\n--- Per-block dQ ---")
for b in range(2):
    k_block = Ks[0,0,b*32:(b+1)*32,:]
    F_block = F_from_O[:, b*32:(b+1)*32]
    dQ_py = F_block @ k_block
    # Can't easily get per-block dQ from kernel, skip
    print(f"  Block {b}: F range [{F_block.min():.4f}, {F_block.max():.4f}], "
          f"K range [{k_block.min():.4f}, {k_block.max():.4f}]")

# The F values should be IDENTICAL between block-by-block and full
# because sigma(s-m) is the same function applied elementwise
# and x_stable = s - m (same m)

print(f"\nDone.")
