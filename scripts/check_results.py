"""Check erp1 + erp2 results on remote server."""
import paramiko, json, sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('i-1.gpushare.com', port=59010, username='root',
               password='HMAV6TEcCARYegYCFEGB6B89sDqqSGfU', timeout=15)

# 1. erp1 log
stdin, stdout, stderr = client.exec_command('cat /hy-tmp/dimsum/outputs/base-erp1/log.txt')
print("=== erp1 结果 ===")
print(stdout.read().decode().strip())

# 2. erp2 tau_star summary
stdin, stdout, stderr = client.exec_command('cat /hy-tmp/dimsum/outputs/erp2_tau_star.json')
raw = stdout.read().decode()
if not raw:
    print("\n=== erp2: NOT FOUND ===")
    client.close()
    sys.exit(1)

data = json.loads(raw)
enc = [m for m in data if 'encoder.layers' in m['module_name']]
dec_self = [m for m in data if 'decoder.layers' in m['module_name'] and 'self_attn' in m['module_name']]
dec_cross = [m for m in data if 'decoder.layers' in m['module_name'] and 'encoder_attn' in m['module_name']]

print("\n=== erp2 tau* 汇总 ===")
for label, arr in [('encoder self', enc), ('decoder self', dec_self), ('decoder cross', dec_cross)]:
    taus = [x for m in arr for x in m['tau_per_head']]
    if taus:
        mean_v = sum(taus)/len(taus)
        min_v = min(taus)
        max_v = max(taus)
        print(f'  {label}: mean={mean_v:.2f}, min={min_v:.2f}, max={max_v:.2f}')

print(f'\n  Total modules: {len(data)}')
print(f'  Total heads: {sum(m["num_heads"] for m in data)}')

# 3. File sizes
stdin, stdout, stderr = client.exec_command('ls -lh /hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt')
fsize = stdout.read().decode().strip()
print(f'\n  erp1 model: {fsize.split()[4] if fsize else "?"}')

client.close()
