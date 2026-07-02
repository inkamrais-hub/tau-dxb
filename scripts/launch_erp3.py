"""Launch erp3 training on remote."""
import paramiko, os

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('i-1.gpushare.com', port=59010, username='root',
               password='HMAV6TEcCARYegYCFEGB6B89sDqqSGfU', timeout=15)

# Make sure train_erp3.py is uploaded
sftp = client.open_sftp()
sftp.put(r'F:\τ\点心杯\scripts\train_erp3.py', '/hy-tmp/dimsum/scripts/train_erp3.py')
print("uploaded train_erp3.py")
sftp.close()

# Kill any existing training
stdin, stdout, stderr = client.exec_command('pkill -f train_erp3 2>/dev/null; sleep 1; echo killed')
print(stdout.read().decode().strip())

# Launch erp3
cmd = (
    'cd /hy-tmp/dimsum && '
    'nohup python3 scripts/train_erp3.py > /hy-tmp/dimsum/erp3.log 2>&1 &'
    'echo "PID=$!"'
)
stdin, stdout, stderr = client.exec_command(cmd)
print(stdout.read().decode().strip())

# Wait a bit and check if it started
import time
time.sleep(5)
stdin, stdout, stderr = client.exec_command('head -20 /hy-tmp/dimsum/erp3.log 2>/dev/null')
print("\n=== erp3.log (first 20 lines) ===")
print(stdout.read().decode())

client.close()
