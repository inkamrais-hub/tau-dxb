"""Upload Triton kernel and test on remote."""
import paramiko
import time

HOST = 'i-1.gpushare.com'
PORT = 59010
USER = 'root'
PASS = 'HMAV6TEcCARYegYCFEGB6B89sDqqSGfU'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Upload
sftp = client.open_sftp()
sftp.put(r'F:\τ\点心杯\scripts\stau_opus_triton.py', '/hy-tmp/dimsum/scripts/stau_opus_triton.py')
sftp.close()
print("uploaded stau_opus_triton.py")

# Check triton
stdin, stdout, stderr = client.exec_command('python3 -c "import triton; print(triton.__version__)" 2>&1', timeout=30)
out = stdout.read().decode()
err = stderr.read().decode()
print("triton check:", out.strip() or err.strip())

# Run smoke test (small first)
print("\nRunning smoke test (small, 1 layer)...")
cmd = (
    'cd /hy-tmp/dimsum && python3 -c "'
    'import sys; sys.path.insert(0, \"scripts\"); '
    'from stau_opus_triton import STauOpusTritonFunction; '
    'import torch; '
    'B,H,M,N,D=2,4,128,128,64; '
    'Q=torch.randn(B,H,M,D,device=\\\"cuda\\\",dtype=torch.float32); '
    'K=torch.randn(B,H,N,D,device=\\\"cuda\\\",dtype=torch.float32); '
    'V=torch.randn(B,H,N,D,device=\\\"cuda\\\",dtype=torch.float32); '
    'lt=torch.zeros(H,device=\\\"cuda\\\",requires_grad=True); '
    'la=torch.zeros(H,device=\\\"cuda\\\",requires_grad=True); '
    'out,attn=STauOpusTritonFunction.apply(Q,K,V,lt,la,\\\"softplus\\\"); '
    'loss=out.sum(); loss.backward(); '
    'print(\\\"FWD\\\",out.shape,\\\"BWD OK\\\"); '
    'print(\\\"NaN\\\",torch.isnan(Q.grad).any().item()); '
    'print(\\\"Inf\\\",torch.isinf(Q.grad).any().item()); '
    'print(\\\"lt_grad\\\",lt.grad.abs().mean().item()); '
    'print(\\\"la_grad\\\",la.grad.abs().mean().item()); '
    '" 2>&1'
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
print(stdout.read().decode())
print(stderr.read().decode())

# Speed test (B=4, H=8, L=1500)
print("Speed test (B=4, H=8, L=1500, D=64)...")
cmd_speed = (
    'cd /hy-tmp/dimsum && python3 -c "'
    'import sys; sys.path.insert(0, \"scripts\"); '
    'from stau_opus_triton import STauOpusTritonFunction; '
    'import torch, time; '
    'B,H,L,D=4,8,1500,64; '
    'Q=torch.randn(B,H,L,D,device=\\\"cuda\\\"); '
    'K=torch.randn(B,H,L,D,device=\\\"cuda\\\"); '
    'V=torch.randn(B,H,L,D,device=\\\"cuda\\\"); '
    'lt=torch.zeros(H,device=\\\"cuda\\\",requires_grad=True); '
    'la=torch.zeros(H,device=\\\"cuda\\\",requires_grad=True); '
    'for _ in range(3): '
    '  o,_=STauOpusTritonFunction.apply(Q,K,V,lt,la,\\\"softplus\\\"); o.sum().backward(); '
    'torch.cuda.synchronize(); t0=time.time(); '
    'for _ in range(10): '
    '  o,_=STauOpusTritonFunction.apply(Q,K,V,lt,la,\\\"softplus\\\"); o.sum().backward(); '
    'torch.cuda.synchronize(); t=(time.time()-t0)/10; '
    'print(\\\"Triton kernel:\\\", round(t*1000,1), \\\"ms/layer\\\"); '
    '" 2>&1'
)
stdin, stdout, stderr = client.exec_command(cmd_speed, timeout=120)
print(stdout.read().decode())
print(stderr.read().decode())

# Compare with original -- speed test on the old operator
print("Speed test (old-style PyTorch operator)...")
cmd_old = (
    'cd /hy-tmp/dimsum && python3 -c "'
    'import sys; sys.path.insert(0, \"scripts\"); '
    'from stau_opus_operator import STauOpusAttentionFunction; '
    'import torch, time; '
    'B,H,L,D=4,8,1500,64; '
    'Q=torch.randn(B,H,L,D,device=\\\"cuda\\\"); '
    'K=torch.randn(B,H,L,D,device=\\\"cuda\\\"); '
    'V=torch.randn(B,H,L,D,device=\\\"cuda\\\"); '
    'lt=torch.zeros(H,device=\\\"cuda\\\",requires_grad=True); '
    'la=torch.zeros(H,device=\\\"cuda\\\",requires_grad=True); '
    'for _ in range(3): '
    '  o,_=STauOpusAttentionFunction.apply(Q,K,V,lt,la,\\\"softplus\\\",None,None); o.sum().backward(); '
    'torch.cuda.synchronize(); t0=time.time(); '
    'for _ in range(10): '
    '  o,_=STauOpusAttentionFunction.apply(Q,K,V,lt,la,\\\"softplus\\\",None,None); o.sum().backward(); '
    'torch.cuda.synchronize(); t=(time.time()-t0)/10; '
    'print(\\\"Old operator:\\\", round(t*1000,1), \\\"ms/layer\\\"); '
    '" 2>&1'
)
stdin, stdout, stderr = client.exec_command(cmd_old, timeout=120)
print(stdout.read().decode())
print(stderr.read().decode())

client.close()
print("\nDone")
