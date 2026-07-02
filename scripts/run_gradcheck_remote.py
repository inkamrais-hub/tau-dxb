"""Upload and run test_gradcheck_tauopus.py on the remote server."""
import paramiko, base64, os

HOST = 'i-1.gpushare.com'
PORT = 59010
USER = 'root'
PASS = 'HMAV6TEcCARYegYCFEGB6B89sDqqSGfU'

local = r'F:\τ\点心杯\scripts\test_gradcheck_tauopus.py'

with open(local, 'rb') as f:
    content = f.read()
b64 = base64.b64encode(content).decode()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Upload via base64
cmd = f"python3 -c \"import base64; open('/hy-tmp/dimsum/scripts/test_gradcheck_tauopus.py','wb').write(base64.b64decode('{b64}'))\""
stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
err = stderr.read().decode().strip()
print(f'uploaded ({len(content)} bytes)' + (f' ERR: {err[:200]}' if err else ''))

# Run
stdin, stdout, stderr = client.exec_command(
    'cd /hy-tmp/dimsum/scripts && python test_gradcheck_tauopus.py 2>&1', timeout=120)
out = stdout.read().decode()
err = stderr.read().decode().strip()
print(out[-3000:] if len(out) > 3000 else out)
if err:
    print('STDERR:', err[:500])

client.close()
