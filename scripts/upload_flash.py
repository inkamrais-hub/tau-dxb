"""Upload flash τ-opus kernel and test."""
import paramiko, sys

HOST = 'i-1.gpushare.com'
PORT = 59010
USER = 'root'
PASS = 'HMAV6TEcCARYegYCFEGB6B89sDqqSGfU'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

sftp = client.open_sftp()
sftp.put(r'F:\τ\点心杯\scripts\stau_opus_flash.py', '/hy-tmp/dimsum/scripts/stau_opus_flash.py')
sftp.close()
print("uploaded")

# Write test script on remote
test_code = '''
import sys
sys.path.insert(0, 'scripts')
import torch
from stau_opus_flash import STauOpusFlashFunction
from stau_opus_operator import STauOpusAttentionFunction
import time

B, H, M, D = 2, 4, 128, 64
Q = torch.randn(B, H, M, D, device='cuda', requires_grad=True)
K = torch.randn(B, H, M, D, device='cuda', requires_grad=True)
V = torch.randn(B, H, M, D, device='cuda', requires_grad=True)
lt = torch.zeros(H, device='cuda', requires_grad=True)
la = torch.zeros(H, device='cuda', requires_grad=True)

out, m = STauOpusFlashFunction.apply(Q, K, V, lt, la, 'softplus')
loss = out.sum()
loss.backward()
print(f"Flash OK | NaN={torch.isnan(Q.grad).any().item()} Inf={torch.isinf(Q.grad).any().item()} lt_g={lt.grad.abs().mean().item():.4f}")

# Comparison with reference
Qr = Q.detach().clone().requires_grad_(True)
Kr = K.detach().clone().requires_grad_(True)
Vr = V.detach().clone().requires_grad_(True)
ltr = torch.zeros(H, device='cuda', requires_grad=True)
lar = torch.zeros(H, device='cuda', requires_grad=True)

out_r, _ = STauOpusAttentionFunction.apply(Qr, Kr, Vr, ltr, lar, 'softplus', None, None)
out_r.sum().backward()
print(f"Ref OK | out_diff={ (out - out_r).abs().max().item():.2e} Q_grad_diff={ (Q.grad - Qr.grad).abs().max().item():.2e}")

# Speed test
B2, H2, L2 = 4, 8, 1500
Q2 = torch.randn(B2, H2, L2, 64, device='cuda', requires_grad=True)
K2 = torch.randn(B2, H2, L2, 64, device='cuda', requires_grad=True)
V2 = torch.randn(B2, H2, L2, 64, device='cuda', requires_grad=True)
lt2 = torch.zeros(H2, device='cuda', requires_grad=True)
la2 = torch.zeros(H2, device='cuda', requires_grad=True)
lt2o = torch.zeros(H2, device='cuda', requires_grad=True)
la2o = torch.zeros(H2, device='cuda', requires_grad=True)

for _ in range(3):
    STauOpusFlashFunction.apply(Q2, K2, V2, lt2, la2, 'softplus')[0].sum().backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    STauOpusFlashFunction.apply(Q2, K2, V2, lt2, la2, 'softplus')[0].sum().backward()
torch.cuda.synchronize()
t_f = (time.time() - t0) / 10
print(f"Flash: {t_f*1000:.0f}ms/layer")

for _ in range(3):
    STauOpusAttentionFunction.apply(Q2, K2, V2, lt2o, la2o, 'softplus', None, None)[0].sum().backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    STauOpusAttentionFunction.apply(Q2, K2, V2, lt2o, la2o, 'softplus', None, None)[0].sum().backward()
torch.cuda.synchronize()
t_o = (time.time() - t0) / 10
print(f"Old:   {t_o*1000:.0f}ms/layer")
print(f"Speedup: {t_o/t_f:.1f}x")

# 12 layers
print()
def run_12(impl, *args):
    total = 0
    Q3 = torch.randn(B2, H2, L2, 64, device='cuda', requires_grad=True)
    K3 = torch.randn(B2, H2, L2, 64, device='cuda', requires_grad=True)
    V3 = torch.randn(B2, H2, L2, 64, device='cuda', requires_grad=True)
    for _ in range(12):
        total += impl(Q3, K3, V3, *args)[0].sum()
    return total

def flash_fn(q, k, v, lt, la):
    return STauOpusFlashFunction.apply(q, k, v, lt, la, 'softplus')

def old_fn(q, k, v, lt, la):
    return STauOpusAttentionFunction.apply(q, k, v, lt, la, 'softplus', None, None)

for i in range(2):
    run_12(flash_fn, lt2, la2).backward()
torch.cuda.synchronize()
t0 = time.time()
for i in range(3):
    run_12(flash_fn, lt2, la2).backward()
torch.cuda.synchronize()
t12_f = (time.time() - t0) / 3
print(f"Flash 12-layers: {t12_f*1000:.0f}ms total")

for i in range(2):
    run_12(old_fn, lt2o, la2o).backward()
torch.cuda.synchronize()
t0 = time.time()
for i in range(3):
    run_12(old_fn, lt2o, la2o).backward()
torch.cuda.synchronize()
t12_o = (time.time() - t0) / 3
print(f"Old 12-layers: {t12_o*1000:.0f}ms total")
print(f"Speedup 12-layers: {t12_o/t12_f:.1f}x")
'''

stdin, stdout, stderr = client.exec_command(f'cat > /hy-tmp/dimsum/scripts/test_flash.py << '"'"'SCRIPT'"'"'\n{test_code}\n'"'"'SCRIPT'"'"'', timeout=30)
stdin, stdout, stderr = client.exec_command('cd /hy-tmp/dimsum && python3 scripts/test_flash.py 2>&1', timeout=300)
print(stdout.read().decode())
err = stderr.read().decode().strip()
if err:
    print("ERR:", err[:500])
client.close()
