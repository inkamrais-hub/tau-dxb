"""
远程 GPU 服务器管理脚本（模板）
复制为 remote.py 并填写服务器信息后使用。
"""
import subprocess, sys, os

HOST = "your-server.gpushare.com"
PORT = 59010
USER = "root"
PASSWORD = "your-password"

BASE_REMOTE = "/hy-tmp/dimsum"
LOCAL_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")

def run_cmd(cmd, capture=True):
    """执行本地命令"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if capture:
        print(result.stdout)
        if result.returncode != 0:
            print(f"  (exit={result.returncode})")
    return result

SSH = f'sshpass -p "{PASSWORD}" ssh -o StrictHostKeyChecking=no -p {PORT} {USER}@{HOST}'
SCP = f'sshpass -p "{PASSWORD}" scp -o StrictHostKeyChecking=no -P {PORT}'

def exec_remote(cmd):
    """在远程执行命令"""
    full_cmd = f'{SSH} "{cmd}"'
    print(f"[remote] {cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.stdout: print(result.stdout)
    if result.stderr: print(result.stderr)
    if result.returncode != 0: print(f"  (exit={result.returncode})")
    return result

def upload(local, remote):
    """上传文件到远程"""
    full_cmd = f'{SCP} {local} {USER}@{HOST}:{remote}'
    print(f"[upload] {local} -> {remote}")
    subprocess.run(full_cmd, shell=True)

def download(remote, local):
    """从远程下载文件"""
    full_cmd = f'{SCP} {USER}@{HOST}:{remote} {local}'
    print(f"[download] {remote} -> {local}")
    subprocess.run(full_cmd, shell=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python remote.py <command> [args]")
        print("命令: exec <cmd> | upload <local> <remote> | download <remote> <local> | status | logs [lines] | watch")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == "exec":
        exec_remote(" ".join(sys.argv[2:]))
    elif cmd == "upload":
        upload(sys.argv[2], sys.argv[3])
    elif cmd == "download":
        download(sys.argv[2], sys.argv[3])
    elif cmd == "status":
        exec_remote(f"tail -5 {BASE_REMOTE}/erp3.log && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader")
    elif cmd == "logs":
        n = sys.argv[2] if len(sys.argv) > 2 else "30"
        exec_remote(f"tail -{n} {BASE_REMOTE}/erp3.log")
    elif cmd == "watch":
        exec_remote(f"watch -n 2 'tail -10 {BASE_REMOTE}/erp3.log && echo --- && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader'")
    else:
        print(f"未知命令: {cmd}")
