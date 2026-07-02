"""Remote test for Triton τ-opus kernel."""
import sys
sys.path.insert(0, 'scripts')
import torch
from stau_opus_triton import STauOpusTritonFunction
from stau_opus_operator import STauOpusAttentionFunction
import time

print("=" * 50)
print("Triton τ-opus kernel test")
print("=" * 50)

# ── small correctness test ──
B, H, M, N, D = 2, 4, 128, 128, 64
Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32, requires_grad=True)
K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
lt = torch.zeros(H, device='cuda', requires_grad=True)
la = torch.zeros(H, device='cuda', requires_grad=True)

out, attn = STauOpusTritonFunction.apply(Q, K, V, lt, la, 'softplus')
loss = out.sum()
loss.backward()

print(f"Forward: {out.shape}, Backward: OK")
print(f"NaN: {torch.isnan(Q.grad).any().item()}")
print(f"Inf: {torch.isinf(Q.grad).any().item()}")
print(f"lt_grad mean: {lt.grad.abs().mean().item():.6f}")
print(f"la_grad mean: {la.grad.abs().mean().item():.6f}")

# ── Compare with reference (small) ──
Qd = Q.detach().clone().requires_grad_(True)
Kd = K.detach().clone().requires_grad_(True)
Vd = V.detach().clone().requires_grad_(True)
ltd = lt.detach().clone().requires_grad_(True)
lad = la.detach().clone().requires_grad_(True)

out_ref, _ = STauOpusAttentionFunction.apply(Qd, Kd, Vd, ltd, lad, 'softplus', None, None)
loss_ref = out_ref.sum()
loss_ref.backward()

print(f"\nReference vs Triton:")
print(f"  Output diff: {(out - out_ref).abs().max().item():.6e}")
print(f"  Q grad diff: {(Q.grad - Qd.grad).abs().max().item():.6e}")
print(f"  lt grad diff: {(lt.grad - ltd.grad).abs().max().item():.6e}")

# ── Speed test: single layer ──
B2, H2, L2, D2 = 4, 8, 1500, 64
Q2 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
K2 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
V2 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
lt2 = torch.zeros(H2, device='cuda', requires_grad=True)
la2 = torch.zeros(H2, device='cuda', requires_grad=True)

# Warmup
for _ in range(3):
    o, _ = STauOpusTritonFunction.apply(Q2, K2, V2, lt2, la2, 'softplus')
    o.sum().backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(10):
    o, _ = STauOpusTritonFunction.apply(Q2, K2, V2, lt2, la2, 'softplus')
    o.sum().backward()
torch.cuda.synchronize()
t_triton = (time.time() - t0) / 10
print(f"\nTriton kernel: {t_triton*1000:.1f}ms/layer (B=4,H=8,L=1500)")

# Old operator
lt2o = torch.zeros(H2, device='cuda', requires_grad=True)
la2o = torch.zeros(H2, device='cuda', requires_grad=True)
for _ in range(3):
    o, _ = STauOpusAttentionFunction.apply(Q2, K2, V2, lt2o, la2o, 'softplus', None, None)
    o.sum().backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(10):
    o, _ = STauOpusAttentionFunction.apply(Q2, K2, V2, lt2o, la2o, 'softplus', None, None)
    o.sum().backward()
torch.cuda.synchronize()
t_old = (time.time() - t0) / 10
print(f"Old operator:  {t_old*1000:.1f}ms/layer (B=4,H=8,L=1500)")
print(f"Speedup: {t_old/t_triton:.1f}x")

# ── Multiple layers (simulate 12 encoder layers) ──
print(f"\nSimulating 12 encoder layers...")
# Triton: each layer does its own Q@K^T, normalization, attn@V
Q3 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
K3 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)
V3 = torch.randn(B2, H2, L2, D2, device='cuda', requires_grad=True)

# Use same params for all 12 layers
def run_12_layers_triton():
    total = 0
    for _ in range(12):
        o, _ = STauOpusTritonFunction.apply(Q3, K3, V3, lt2, la2, 'softplus')
        total += o.sum()
    return total

def run_12_layers_old():
    total = 0
    for _ in range(12):
        o, _ = STauOpusAttentionFunction.apply(Q3, K3, V3, lt2o, la2o, 'softplus', None, None)
        total += o.sum()
    return total

# Warmup
for _ in range(2):
    run_12_layers_triton().backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(3):
    run_12_layers_triton().backward()
torch.cuda.synchronize()
t12_triton = (time.time() - t0) / 3
print(f"Triton 12 layers: {t12_triton*1000:.0f}ms total ({t12_triton/12*1000:.1f}ms/layer)")

for _ in range(2):
    run_12_layers_old().backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(3):
    run_12_layers_old().backward()
torch.cuda.synchronize()
t12_old = (time.time() - t0) / 3
print(f"Old 12 layers: {t12_old*1000:.0f}ms total ({t12_old/12*1000:.1f}ms/layer)")
print(f"Speedup 12 layers: {t12_old/t12_triton:.1f}x")

print("\nDone.")
