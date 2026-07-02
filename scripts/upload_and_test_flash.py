"""Upload via SSH exec (avoid sftp path issues)."""
import paramiko, sys, os

HOST = 'i-1.gpushare.com'
PORT = 59010
USER = 'root'
PASS = 'HMAV6TEcCARYegYCFEGB6B89sDqqSGfU'

local_dir = r'F:\τ\点心杯\scripts'
remote_dir = '/hy-tmp/dimsum/scripts'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

for fname in ['stau_opus_flash.py', 'test_flash_rt.py']:
    local = os.path.join(local_dir, fname)
    with open(local, 'rb') as f:
        content = f.read()
    # Write via python on remote to avoid shell escaping issues
    safe = content.decode('utf-8').replace("'", "'\\''")
    cmd = f"python3 -c \"import sys; open('{remote_dir}/{fname}','w').write(sys.stdin.read())\" << 'PYEOF'\n{content.decode('utf-8')}\nPYEOF"
    # Too risky with heredoc, use python base64 approach instead
    import base64
    b64 = base64.b64encode(content).decode()
    cmd = f"python3 -c \"import base64; open('{remote_dir}/{fname}','wb').write(base64.b64decode('{b64}'))\""
    stdin, stdout, stderr = client.exec_command(f'mkdir -p {remote_dir} && {cmd}', timeout=30)
    out = stdout.read().decode()
    err = stderr.read().decode().strip()
    print(f"uploaded {fname} ({len(content)} bytes)")
    if err:
        print(f"  ERR: {err[:200]}")
    if out:
        print(f"  OUT: {out[:200]}")

# Verify files exist
stdin, stdout, stderr = client.exec_command(f'ls -la {remote_dir}/stau_opus_flash.py {remote_dir}/test_flash_rt.py 2>&1', timeout=10)
print(stdout.read().decode())
err = stderr.read().decode().strip()
if err:
    print("LS ERR:", err[:500])

# Run test
print("\n--- Running test ---")
stdin, stdout, stderr = client.exec_command(
    'cd /hy-tmp/dimsum && python3 scripts/test_flash_rt.py 2>&1',
    timeout=600
)
print(stdout.read().decode())
err = stderr.read().decode().strip()
if err:
    print("STDERR:", err[:2000])
client.close()
