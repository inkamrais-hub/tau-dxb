"""Upload and run smoke test + erp3 launch."""
import paramiko, os

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('i-1.gpushare.com', port=59010, username='root',
               password='HMAV6TEcCARYegYCFEGB6B89sDqqSGfU', timeout=15)

sftp = client.open_sftp()
sftp.put(r'F:\τ\点心杯\scripts\stau_opus_operator.py', '/hy-tmp/dimsum/scripts/stau_opus_operator.py')
print("uploaded stau_opus_operator.py")
sftp.close()

print("\n=== Running smoke test ===")
stdin, stdout, stderr = client.exec_command(
    'cd /hy-tmp/dimsum && python3 scripts/_smoke_test_erp3.py 2>&1',
    timeout=120
)
print(stdout.read().decode())

client.close()
