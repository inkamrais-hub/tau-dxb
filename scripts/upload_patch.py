"""Upload patch_whisper_stau.py + stau_opus_flash.py to remote server."""
import paramiko, os, base64, time

HOST = 'i-1.gpushare.com'
PORT = 59010
USER = 'root'
PASS = 'HMAV6TEcCARYegYCFEGB6B89sDqqSGfU'

local_dir = r'F:\τ\点心杯\scripts'
remote_dir = '/hy-tmp/dimsum/scripts'

files = ['stau_opus_flash.py', 'patch_whisper_stau.py']

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Upload files
for fname in files:
    local = os.path.join(local_dir, fname)
    with open(local, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode()
    cmd = f"python3 -c \"import base64; open('{remote_dir}/{fname}','wb').write(base64.b64decode('{b64}'))\""
    stdin, stdout, stderr = client.exec_command(f'mkdir -p {remote_dir} && {cmd}', timeout=30)
    err = stderr.read().decode().strip()
    print(f"uploaded {fname} ({len(content)} bytes)" + (f" ERR: {err[:200]}" if err else ""))

# Verify
stdin, stdout, stderr = client.exec_command(f'ls -la {remote_dir}/stau_opus_flash.py {remote_dir}/patch_whisper_stau.py 2>&1', timeout=10)
print(stdout.read().decode())
err = stderr.read().decode().strip()
if err: print("LS ERR:", err[:500])

client.close()
print("Done.")
