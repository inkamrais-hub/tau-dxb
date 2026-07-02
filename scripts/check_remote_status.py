"""Check remote erp1/erp2 outputs and GPU status."""
import paramiko, json

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('i-1.gpushare.com', port=59010, username='root',
               password='HMAV6TEcCARYegYCFEGB6B89sDqqSGfU', timeout=15)

# erp1 log
stdin, stdout, stderr = client.exec_command('cat /hy-tmp/dimsum/outputs/base-erp1/log.txt 2>/dev/null')
print("=== erp1 log ===")
print(stdout.read().decode().strip())

# erp1 model file
stdin, stdout, stderr = client.exec_command('ls -lh /hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt 2>/dev/null')
print("\n=== erp1 best_model ===")
print(stdout.read().decode().strip())

# erp2 tau_star
stdin, stdout, stderr = client.exec_command('cat /hy-tmp/dimsum/outputs/erp2_tau_star.json 2>/dev/null')
raw = stdout.read().decode()
if raw:
    data = json.loads(raw)
    enc = [m for m in data if 'encoder.layers' in m['module_name']]
    dec_self = [m for m in data if 'decoder.layers' in m['module_name'] and 'self_attn' in m['module_name']]
    dec_cross = [m for m in data if 'decoder.layers' in m['module_name'] and 'encoder_attn' in m['module_name']]
    print("\n=== erp2 tau* ===")
    for label, arr in [('encoder self', enc), ('decoder self', dec_self), ('decoder cross', dec_cross)]:
        taus = [x for m in arr for x in m['tau_per_head']]
        if taus:
            print(f"  {label}: mean={sum(taus)/len(taus):.2f}, min={min(taus):.2f}, max={max(taus):.2f}")
    print(f"  Total modules: {len(data)}, heads: {sum(m['num_heads'] for m in data)}")
else:
    print("\n=== erp2: NOT FOUND ===")

# GPU
stdin, stdout, stderr = client.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader')
print("\n=== GPU ===")
print(stdout.read().decode().strip())

# running processes
stdin, stdout, stderr = client.exec_command('ps aux | grep -E "train_|compute_tau" | grep -v grep')
print("\n=== running processes ===")
print(stdout.read().decode().strip() or "(none)")

client.close()
