"""Upload and run full erp3 smoke test."""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('i-1.gpushare.com', port=59010, username='root',
               password='HMAV6TEcCARYegYCFEGB6B89sDqqSGfU', timeout=15)

sftp = client.open_sftp()
sftp.put(r'F:\τ\点心杯\scripts\_smoke_erp3_full.py', '/hy-tmp/dimsum/scripts/_smoke_erp3_full.py')
print("uploaded _smoke_erp3_full.py")
sftp.close()

print("\n=== Running full smoke test ===")
stdin, stdout, stderr = client.exec_command(
    'cd /hy-tmp/dimsum && python3 scripts/_smoke_erp3_full.py 2>&1',
    timeout=180
)
print(stdout.read().decode())

client.close()
