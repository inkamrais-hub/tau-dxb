"""Compare Triton dQ kernel output vs manual computation block by block."""
import sys, time
sys.path.insert(0, 'scripts')
import torch
import triton
from stau_opus_flash import (STauOpusFlashFunction, _stau_opus_flash_fwd, 
                               _stau_opus_flash_bwd_dq, _stau_opus_flash_bwd_dkdv)
from stau_opus_operator import STauOpusAttentionFunction

torch.manual_seed(42)
B, H, L, D = 1, 1, 128, 64

Q = torch.randn(B, H, L, D, device='cuda')
K = torch.randn(B, H, L, D, device='cuda')
V = torch.randn(B, H, L, D, device='cuda')
dO = torch.randn(B, H, L, D, device='cuda')
lt = torch.zeros(H, device='cuda', requires_grad=True)
la = torch.zeros(H, device='cuda', requires_grad=True)

scale = float(D ** -0.5)
Qf = Q.float()
Kf = K.float()
Vf = V.float()
Qs = Qf * (scale ** 0.5)
Ks = Kf * (scale ** 0.5)
tau = (torch.nn.functional.softplus(lt) + 1.0).float()
alpha = torch.exp(la).float()

# Run flash forward (with argmax)
BLOCK_M, BLOCK_N = 32, 64
m_buf = torch.empty(B, H, L, device='cuda', dtype=torch.float32)
l_buf = torch.empty(B, H, L, device='cuda', dtype=torch.float32)
argmax_buf = torch.empty(B, H, L, device='cuda', dtype=torch.int32)
O_buf = torch.empty_like(Qs)
num_m = triton.cdiv(L, BLOCK_M)
num_n = triton.cdiv(L, BLOCK_N)

grid = (B, H, num_m)
_stau_opus_flash_fwd[grid](
    Qs, Ks, Vf, O_buf, m_buf, l_buf, argmax_buf,
    Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
    Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
    Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
    O_buf.stride(0), O_buf.stride(1), O_buf.stride(2), O_buf.stride(3),
    m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
    l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
    argmax_buf.stride(0), argmax_buf.stride(1), argmax_buf.stride(2),
    tau, alpha, B, H, L, D,
    BLOCK_M=32, BLOCK_N=64, BLOCK_D=D, sigma_name='softplus',
)

# Compute dQ directly with Triton kernel
dQ_triton = torch.empty_like(Qs)
dK_triton = torch.empty_like(Ks)
dV_triton = torch.empty_like(Vf)
total_F = torch.empty(B, H, L, device='cuda', dtype=torch.float32)  # NEW
dO_f = dO.float()

grid_dq = (B, H, num_m)
_stau_opus_flash_bwd_dq[grid_dq](
    Qs, Ks, Vf, dO_f, O_buf, dQ_triton,
    m_buf, l_buf, total_F,  # NEW: total_F
    Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
    Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
    Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
    dO_f.stride(0), dO_f.stride(1), dO_f.stride(2), dO_f.stride(3),
    O_buf.stride(0), O_buf.stride(1), O_buf.stride(2), O_buf.stride(3),
    dQ_triton.stride(0), dQ_triton.stride(1), dQ_triton.stride(2), dQ_triton.stride(3),
    m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
    l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
    total_F.stride(0), total_F.stride(1), total_F.stride(2),
    tau, alpha, B, H, L, D,
    BLOCK_M=32, BLOCK_N=64, BLOCK_D=D, sigma_name='softplus',
)

grid_dkdv = (B, H, num_n)
_stau_opus_flash_bwd_dkdv[grid_dkdv](
    Qs, Ks, Vf, dO_f, O_buf, dK_triton, dV_triton, m_buf, l_buf,
    Qs.stride(0), Qs.stride(1), Qs.stride(2), Qs.stride(3),
    Ks.stride(0), Ks.stride(1), Ks.stride(2), Ks.stride(3),
    Vf.stride(0), Vf.stride(1), Vf.stride(2), Vf.stride(3),
    dO_f.stride(0), dO_f.stride(1), dO_f.stride(2), dO_f.stride(3),
    O_buf.stride(0), O_buf.stride(1), O_buf.stride(2), O_buf.stride(3),
    dK_triton.stride(0), dK_triton.stride(1), dK_triton.stride(2), dK_triton.stride(3),
    dV_triton.stride(0), dV_triton.stride(1), dV_triton.stride(2), dV_triton.stride(3),
    m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
    l_buf.stride(0), l_buf.stride(1), l_buf.stride(2),
    tau, alpha, B, H, L, D,
    BLOCK_M=32, BLOCK_N=64, BLOCK_D=D, sigma_name='softplus',
)

# ── Apply global argmax correction (same as wrapper) ──
argmax_i64 = argmax_buf.long()
argmax_expanded = argmax_i64.unsqueeze(-1).expand(-1, -1, -1, D)
K_at_argmax = torch.gather(Ks, dim=2, index=argmax_expanded)
dQ_correction = total_F.unsqueeze(-1) * K_at_argmax
dQ_triton.sub_(dQ_correction)

Q_times_totalF = total_F.unsqueeze(-1) * Qs
dK_triton.scatter_add_(dim=2, index=argmax_expanded, src=-Q_times_totalF)

# dQ_triton is now dL/dQs (gradient w.r.t. Qs = Q*sqrt(scale))
# dQ_correct = dQ_triton * sqrt(scale) = dL/dQ
dQ_correct = dQ_triton * (scale ** 0.5)
dK_correct = dK_triton * (scale ** 0.5)

# Reference from API
Qr = Q.detach().clone().requires_grad_(True)
Kr = K.detach().clone().requires_grad_(True)
Vr = V.detach().clone().requires_grad_(True)
out_r, _ = STauOpusAttentionFunction.apply(Qr, Kr, Vr, lt.detach().clone(), la.detach().clone(), 'softplus', None, None)
out_r.backward(dO)

# Reference dQ from manual formula
scores = Qf[0,0] @ Kf[0,0].T * scale
m = scores.max(dim=-1, keepdim=True).values
x = scores - m
sigma = torch.nn.functional.softplus(x * alpha[0].item())
log_q = tau[0].item() * sigma.clamp(min=1e-8).log()
q = log_q.exp()
l = q.sum(dim=-1, keepdim=True)
attn = q / l.clamp(min=1e-8)
dP = dO[0,0] @ Vf[0,0].T
weighted = (attn * dP).sum(dim=-1, keepdim=True)
d_log_sigma = tau[0].item() * attn * (dP - weighted)
sigma_p = torch.sigmoid(x * alpha[0].item())
F = d_log_sigma * (sigma_p / sigma.clamp(min=1e-8)) * alpha[0].item()
# CORRECT argmax correction: global sum
total_F_manual = F.sum(dim=-1, keepdim=True)  # global sum over all positions
is_max = (x.abs() < 1e-6)
d_s = torch.where(is_max, F - total_F_manual, F)
dQ_manual = (d_s @ Kf[0,0] * scale).unsqueeze(0).unsqueeze(0)

# dK manual
dK_manual = (F.T @ Qf[0,0] * scale).unsqueeze(0).unsqueeze(0)
# Apply correction: dK[argmax_j,:] -= total_F[j] * Q[j,:]
dK_corrected_manual = dK_manual.clone()
for j in range(L):
    am = scores[0,j].argmax()  # argmax for query j
    dK_corrected_manual[0,0,am,:] -= F[j,am].item() * Qf[0,0,j,:] * scale

print("=" * 60)
print("dQ comparison (after global argmax correction):")
print(f"  dQ_triton (dL/dQs) range: [{dQ_triton.min():.4f}, {dQ_triton.max():.4f}]")
dq_expected = d_s @ Ks[0,0]
print(f"  dQ_expected (d_s@Ks) range: [{dq_expected.min():.4f}, {dq_expected.max():.4f}]")
print(f"  dQ_manual     range: [{dQ_manual.min():.4f}, {dQ_manual.max():.4f}]")
print(f"  dQ_ref API    range: [{Qr.grad.min():.4f}, {Qr.grad.max():.4f}]")

# Now dQ_triton (corrected) should match d_s @ Ks
dQ_expected_raw = (d_s @ Ks[0,0]).unsqueeze(0).unsqueeze(0)
diff_dq = (dQ_triton - dQ_expected_raw).abs().max().item()
print(f"\n  Triton corrected vs expected (d_s@Ks): diff = {diff_dq:.2e}")

# dK comparison
dK_expected_raw = (F.T @ Qs[0,0]).unsqueeze(0).unsqueeze(0)
# Apply same argmax correction
for j in range(L):
    am = scores[0,j].argmax()
    dK_expected_raw[0,0,am,:] -= total_F[0,0,j].item() * Qs[0,0,j,:]
diff_dk = (dK_triton - dK_expected_raw).abs().max().item()
print(f"  dK_triton dL/dKs vs expected: diff = {diff_dk:.2e}")

# Compare with reference API (after scale correction)
print(f"\n  dQ_correct vs ref API: diff = {(dQ_correct - Qr.grad).abs().max().item():.2e}")
print(f"  dK_correct vs ref API: diff = {(dK_correct - Kr.grad).abs().max().item():.2e}")
print(f"  dV vs ref API: diff = {(dV_triton - Vr.grad).abs().max().item():.2e}")

# ── Deep diagnostic: check intermediate values ──
q_tile = Qs[0,0, :32, :]
k_tile0 = Ks[0,0, :64, :]
k_tile1 = Ks[0,0, 64:, :]
v_tile0 = Vf[0,0, :64, :]
v_tile1 = Vf[0,0, 64:, :]
do_tile = dO_f[0,0, :32, :]
m_row = m_buf[0,0, :32]
l_row = l_buf[0,0, :32]
inv_l = 1.0 / l_row.clamp(min=1e-8)

def compute_F_block(q, k, v, do, m_row, inv_l, tau_v, alpha_v, weighted_global):
    s = q @ k.T
    x = s - m_row.unsqueeze(-1)
    sigma = torch.nn.functional.softplus(x * alpha_v)
    log_q_v = tau_v * sigma.clamp(min=1e-8).log()
    q_v = log_q_v.exp()
    attn = q_v * inv_l.unsqueeze(-1)
    dP = do @ v.T
    d_log_sigma = tau_v * attn * (dP - weighted_global)  # GLOBAL weighted
    sigma_p = torch.sigmoid(x * alpha_v)
    F_block = d_log_sigma * (sigma_p / sigma.clamp(min=1e-8)) * alpha_v
    return s, x, sigma, attn, dP, d_log_sigma, sigma_p, F_block

# Compute global weighted from forward output O
weighted_global_0 = (do_tile * O_buf[0,0,:32,:]).sum(dim=-1, keepdim=True)

s0, x0, sigma0, attn0, dP0, dls0, sp0, F0 = compute_F_block(q_tile, k_tile0, v_tile0, do_tile, m_row, inv_l, tau[0].item(), alpha[0].item(), weighted_global_0)
s1, x1, sigma1, attn1, dP1, dls1, sp1, F1 = compute_F_block(q_tile, k_tile1, v_tile1, do_tile, m_row, inv_l, tau[0].item(), alpha[0].item(), weighted_global_0)

dQ_q0_triton = dQ_triton[0,0, :32, :]
dQ_q0_manual = F0 @ k_tile0 + F1 @ k_tile1  # Σ Fi @ Ki (no correction)

print(f"\n--- Q-block 0 verification (before global correction) ---")
print(f"  dQ (Σ Fi@Ki) manual vs triton: diff = {(dQ_q0_manual - dQ_q0_triton).abs().max().item():.2e}")
print(f"  total_F manual vs triton: diff = {(F0.sum()+F1.sum() - total_F[0,0,:32]).abs().max().item():.2e}")
print(f"  total_F ranges:")
print(f"    manual block0 F sum: {F0.sum().item():.4f}, manual total (F0+F1): {(F0+F1).sum().item():.4f}")
print(f"    triton total_F: [{total_F[0,0,:32].min():.4f}, {total_F[0,0,:32].max():.4f}]")

# Check F0 row-by-row
print("\n  F values per row (first 8 rows, first 4 cols):")
for row in range(min(8, 32)):
    manual_vals = F0[row, :4]
    # We can't easily extract Triton F values, so just check dQ
    pass

# Check specific cells: row 0 of F0
print(f"  F0[0,:8] manual: {F0[0,:8].tolist()}")
# dQ first row
print(f"  dQ_manual[0,:8]: {dQ_q0_manual[0,:8].tolist()}")
print(f"  dQ_triton[0,:8]: {dQ_q0_triton[0,:8].tolist()}")

# Check dK per-block contribution vs Triton from dK kernel
# dK0 from Triton: K-block 0 (first 64 keys), iterate over all Q-blocks
dK_block0_triton = dK_triton[0,0, :64, :]  # first KV block's dK

# Manual: dK contribution for KV-block 0
# dK = Σ F_i^T @ Q_i over all Q blocks
# For Q-block 0: F0.T @ q_tile
# For Q-block 1: F2.T @ q_tile2...
dK_block0_manual = F0.T @ q_tile  # only Q-block 0, not full

q_tile2 = Qs[0,0, 32:64, :]
# Global weighted for Q-block 1 from forward output O
weighted_global_1 = (dO_f[0,0, 32:64, :] * O_buf[0,0, 32:64, :]).sum(dim=-1, keepdim=True)
s2 = q_tile2 @ k_tile0.T
x2 = s2 - m_buf[0,0, 32:64].unsqueeze(-1)
sigma2 = torch.nn.functional.softplus(x2 * alpha[0].item())
log_q2 = tau[0].item() * sigma2.clamp(min=1e-8).log()
q2 = log_q2.exp()
attn2 = q2 / l_buf[0,0, 32:64].unsqueeze(-1).clamp(min=1e-8)
dP2 = dO_f[0,0, 32:64, :] @ v_tile0.T
dls2 = tau[0].item() * attn2 * (dP2 - weighted_global_1)  # GLOBAL weighted
sp2 = torch.sigmoid(x2 * alpha[0].item())
F2 = dls2 * (sp2 / sigma2.clamp(min=1e-8)) * alpha[0].item()
dK_block0_manual += F2.T @ q_tile2  # add Q-block 1

# Check specific dK values
print(f"\n  dK_block0 manual [0,:8]: {dK_block0_manual[0,:8].tolist()}")
print(f"  dK_block0 triton [0,:8]: {dK_block0_triton[0,:8].tolist()}")
print(f"  dK_block0 manual vs triton diff: {(dK_block0_manual - dK_block0_triton).abs().max().item():.2e}")

# Check: does the kernel catch the same intermediate sigma values?
# We can check by making a mini-test with known inputs
print("\n--- Sanity: intermediate values ---")
print(f"  m_row[:8]: {m_row[:8].tolist()}")
print(f"  l_row[:8]: {l_row[:8].tolist()}")
print(f"  inv_l[:8]: {inv_l[:8].tolist()}")
# Check s0 first row max/min
print(f"  s0[0,:] range: [{s0[0,:].min():.4f}, {s0[0,:].max():.4f}]")
print(f"  sigma0[0,:8]: {sigma0[0,:8].tolist()}")
print(f"  attn0[0,:8]: {attn0[0,:8].tolist()}")
print(f"  dP0[0,:8]: {dP0[0,:8].tolist()}")

# Verify argmax
print(f"\n--- Argmax verification ---")
print(f"  argmax (first 16 queries): {argmax_buf[0,0,:16].tolist()}")
print(f"  manual argmax: {[scores[j].argmax().item() for j in range(16)]}")

print(f"\nDone.")
