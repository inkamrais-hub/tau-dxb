"""
用 paramiko 部署文件到远程服务器并执行命令。
避免 Windows shell 引号/转义问题。
"""
import os
import sys
import io
import stat
import getpass

try:
    import paramiko
except ImportError:
    os.system(f"{sys.executable} -m pip install paramiko -q")
    import paramiko

# ===== 配置 =====
HOST = "i-1.gpushare.com"
PORT = 59010
USER = "root"
PASSWORD = "HMAV6TEcCARYegYCFEGB6B89sDqqSGfU"

LOCAL_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_SCRIPTS_DIR = "/hy-tmp/dimsum/scripts"


def upload_file(sftp, local_path, remote_path):
    """上传单个文件，跳过已存在且大小一致的文件"""
    if not os.path.exists(local_path):
        print(f"  [SKIP] 本地文件不存在: {local_path}")
        return False
    try:
        sftp.stat(remote_path)
        local_size = os.path.getsize(local_path)
        remote_size = sftp.stat(remote_path).st_size
        if local_size == remote_size:
            print(f"  [SKIP] 已存在且大小一致: {remote_path}")
            return False
    except FileNotFoundError:
        pass
    print(f"  [UPLOAD] {local_path} -> {remote_path}")
    sftp.put(local_path, remote_path)
    return True


def upload_file_content(sftp, content, remote_path):
    """直接上传内存中的内容"""
    print(f"  [UPLOAD] {remote_path}")
    with sftp.open(remote_path, "w") as f:
        f.write(content)


def ensure_dir(sftp, path):
    """确保远程目录存在"""
    try:
        sftp.stat(path)
    except FileNotFoundError:
        print(f"  [MKDIR] {path}")
        sftp.mkdir(path)


def exec_remote(ssh, command, verbose=True):
    """在远程执行命令并返回输出"""
    if verbose:
        print(f"  [EXEC] {command[:200]}")
    stdin, stdout, stderr = ssh.exec_command(command, timeout=300)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if verbose and out:
        for line in out.split("\n"):
            print(f"    | {line}")
    if verbose and err and exit_code != 0:
        for line in err.split("\n"):
            print(f"    ! {line}")
    return exit_code, out, err


def fix_patch_sparse_grad(sftp, ssh):
    """修复 patch_whisper_sparse_grad.py 中 WhisperFeedForward 的 import"""
    remote_path = f"{REMOTE_SCRIPTS_DIR}/patch_whisper_sparse_grad.py"
    
    # 读取远程文件
    with sftp.open(remote_path, "rb") as f:
        content = f.read().decode("utf-8")
    
    # 找到 WhisperFeedForward 的引用，替换为 fc1/fc2 的方式
    # Transformers 5.x 移除了 WhisperFeedForward，直接使用 fc1/fc2
    old_import = "from transformers.models.whisper.modeling_whisper import WhisperFeedForward"
    new_import = "# WhisperFeedForward removed in transformers 5.x; fallback to fc1/fc2"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
    
    # 替换 apply_sparse_grad_wta 中引用 WhisperFeedForward 的地方
    # 把 target submodules 中的 WhisperFeedForward 改成空列表（不匹配任何模块）
    old_target = """target_submodules = {
        "encoder+decoder": [WhisperEncoderLayer, WhisperDecoderLayer],
        "encoder": [WhisperEncoderLayer],
        "decoder": [WhisperDecoderLayer],
        "ffn_only": [WhisperFeedForward],
        "ffn_only_encoder": [WhisperFeedForward],
    }"""
    new_target = """target_submodules = {
        "encoder+decoder": [WhisperEncoderLayer, WhisperDecoderLayer],
        "encoder": [WhisperEncoderLayer],
        "decoder": [WhisperDecoderLayer],
        "ffn_only": [],  # WhisperFeedForward removed in transformers 5.x
        "ffn_only_encoder": [],
    }"""
    
    if old_target in content:
        content = content.replace(old_target, new_target)
    
    # 替换 set_sparse_grad_k_ratio 中的 import
    old_import2 = "from transformers.models.whisper.modeling_whisper import WhisperFeedForward"
    if old_import2 in content:
        content = content.replace(old_import2, new_import)
    
    # 替换 set_sparse_grad_k_ratio 中的引用
    old_k_ratio = """def set_sparse_grad_k_ratio(model, k_ratio):
    \"\"\"Set k_ratio for all SparseGradWTA hooks on WhisperFeedForward modules.\"\"\"
    for module in model.modules():
        if isinstance(module, WhisperFeedForward):
            for hook in getattr(module, "_sparse_grad_hooks", []):
                hook.k_ratio = k_ratio"""
    new_k_ratio = """def set_sparse_grad_k_ratio(model, k_ratio):
    \"\"\"Set k_ratio for all SparseGradWTA hooks on ff modules.\"\"\"
    count = 0
    for module in model.modules():
        for hook in getattr(module, "_sparse_grad_hooks", []):
            hook.k_ratio = k_ratio
            count += 1
    if count == 0:
        print(f"[SparseGrad] WARNING: no hooks found for k_ratio update")"""
    
    if old_k_ratio in content:
        content = content.replace(old_k_ratio, new_k_ratio)
    # 写回
    with sftp.open(remote_path, "w") as f:
        f.write(content)
    print(f"  [PATCH] 已修复 {remote_path}")


def main():
    print("=" * 60)
    print(f"连接到 {HOST}:{PORT} ...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, look_for_keys=False, allow_agent=False)
    sftp = ssh.open_sftp()
    print(f"  连接成功！")
    
    # 1. 确保远程目录存在
    ensure_dir(sftp, REMOTE_SCRIPTS_DIR)
    
    # 2. 上传 train_erp1.py
    upload_file(sftp, 
                os.path.join(LOCAL_SCRIPTS_DIR, "train_erp1.py"),
                f"{REMOTE_SCRIPTS_DIR}/train_erp1.py")
    
    # 3. 上传 wait_and_start_erp1.sh
    upload_file(sftp,
                os.path.join(LOCAL_SCRIPTS_DIR, "wait_and_start_erp1.sh"),
                f"{REMOTE_SCRIPTS_DIR}/wait_and_start_erp1.sh")
    
    # 4. 修复 patch_whisper_sparse_grad.py
    fix_patch_sparse_grad(sftp, ssh)
    
    # 5. 确保脚本可执行
    exec_remote(ssh, f"chmod +x {REMOTE_SCRIPTS_DIR}/wait_and_start_erp1.sh")
    
    # 6. 杀掉旧的训练进程
    exec_remote(ssh, "pkill -f train_erp1 2>/dev/null; pkill -f wait_and_start 2>/dev/null; sleep 2; echo 'old processes killed'")
    
    # 7. 验证模型文件存在
    code, out, _ = exec_remote(ssh, "ls -la /hy-tmp/whisper-small-local/model.safetensors")
    if code != 0:
        print("  [ERROR] 模型文件 /hy-tmp/whisper-small-local 不存在！")
        sftp.close()
        ssh.close()
        sys.exit(1)
    model_size = out.split()[4] if out else "?"
    print(f"  [OK] 模型文件确认 ({model_size} bytes)")
    
    # 8. 验证数据文件存在
    code, out, _ = exec_remote(ssh, "wc -l /hy-tmp/dimsum/data/prepared/train.jsonl /hy-tmp/dimsum/data/prepared/val.jsonl 2>/dev/null")
    if code == 0:
        print(f"  [OK] 数据文件确认:\n{out}")
    else:
        print("  [ERROR] 数据文件不存在，需要先运行 prepare_data_remote.py！")
        sftp.close()
        ssh.close()
        sys.exit(1)
    
    # 9. 备份旧的日志
    exec_remote(ssh, "mv /hy-tmp/dimsum/erp1.log /hy-tmp/dimsum/erp1.log.$(date +%s) 2>/dev/null; echo 'log backed up'")
    
    # 10. 启动训练
    print("\n" + "=" * 60)
    print("启动 erp1 训练...")
    cmd = (
        "cd /hy-tmp/dimsum && "
        "nohup bash /hy-tmp/dimsum/scripts/wait_and_start_erp1.sh "
        "> /hy-tmp/dimsum/wait_and_start_erp1.log 2>&1 &"
    )
    _, _, _ = exec_remote(ssh, cmd)
    
    # 11. 等待一会，检查是否启动成功
    import time
    time.sleep(15)
    
    # 12. 检查训练状态
    print("\n" + "=" * 60)
    print("训练状态检查:")
    code, out, _ = exec_remote(ssh, 
        "ps aux | grep train_erp1 | grep -v grep || echo 'NOT RUNNING'")
    
    if 'NOT RUNNING' in out:
        print("  [FAIL] 训练未启动！检查日志:")
        code, out, _ = exec_remote(ssh, "tail -60 /hy-tmp/dimsum/erp1.log 2>/dev/null || echo 'no log'")
    else:
        print(f"  [OK] 训练进程运行中")
        # GPU 状态
        exec_remote(ssh, 
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv")
        # 日志尾巴
        print("  --- 训练日志 (tail -15) ---")
        exec_remote(ssh, "tail -15 /hy-tmp/dimsum/erp1.log")
    
    # 清理
    sftp.close()
    ssh.close()
    print("\n" + "=" * 60)
    print("部署完成！")
    print(f"日志文件: /hy-tmp/dimsum/erp1.log")
    print(f"训练产出: /hy-tmp/dimsum/outputs/base-erp1/")


if __name__ == "__main__":
    main()
