"""
点心杯远程训练长期监控脚本。
实时显示 GPU 状态 + 训练进度，支持自动刷新。

用法：
  python scripts/monitor_local.py              # 默认 2s 刷新
  python scripts/monitor_local.py --interval 5  # 5s 刷新
  python scripts/monitor_local.py --log erp3    # 监控 erp3.log
  python scripts/monitor_local.py --once        # 单次输出，不循环
"""
import argparse
import time
import os
import sys
from datetime import datetime

HOST = "i-1.gpushare.com"
PORT = 59010
USER = "root"
PASSWORD = "HMAV6TEcCARYegYCFEGB6B89sDqqSGfU"
REMOTE_BASE = "/hy-tmp/dimsum"


def get_client():
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    return client


def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out.strip(), err.strip()


def get_gpu_info(client):
    """Get GPU status: index, name, util%, memory, temp."""
    out, _ = run_cmd(client,
        'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu '
        '--format=csv,noheader,nounits', timeout=15)
    lines = out.split("\n")
    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "util": parts[2],
                "mem_used": parts[3],
                "mem_total": parts[4],
                "temp": parts[5],
            })
    return gpus


def get_training_progress(client, log_name="erp4"):
    """Get latest training progress by remote Python extraction."""
    log_path = f"{REMOTE_BASE}/{log_name}.log"

    # Get last tqdm line via remote Python (avoids \r corruption from tail)
    cmd = (
        'python3 -c "'
        'import re; '
        'f=open(\\"' + log_path + '\\",\\"r\\"); '
        'lines=f.readlines(); '
        'f.close(); '
        'tqdm=[l for l in lines if \\"%|\\" in l]; '
        'print(tqdm[-1].strip() if tqdm else \\"NO_TQDM\\")'
        '"'
    )
    tqdm_out, _ = run_cmd(client, cmd, timeout=15)

    # Also get epoch header
    cmd2 = (
        'python3 -c "'
        'import re; '
        'f=open(\\"' + log_path + '\\",\\"r\\"); '
        'lines=f.readlines(); '
        'f.close(); '
        'ep=[l for l in lines if \\"Epoch \\" in l and re.search(\\"Epoch \\\\d+/\\\\d+\\", l)]; '
        'print(ep[-1].strip() if ep else \\"NO_EPOCH\\")'
        '"'
    )
    ep_out, _ = run_cmd(client, cmd2, timeout=15)

    # Check if process is running
    ps_out, _ = run_cmd(client,
        f"ps aux | grep train_{log_name} | grep -v grep || echo 'NOT RUNNING'", timeout=10)

    process_running = "NOT RUNNING" not in ps_out and ps_out.strip() != ""

    epoch = speed = loss = lr = step = pbar_pct = "—"

    # Parse epoch from ep_out
    ep_match = __import__('re').search(r"Epoch (\d+)/(\d+)", ep_out)
    if ep_match:
        epoch = f"{ep_match.group(1)}/{ep_match.group(2)}"

    # Parse tqdm line — this is a single clean line
    if tqdm_out and tqdm_out != "NO_TQDM":
        line = tqdm_out.replace("\r", "")
        pct = __import__('re').search(r"(\d+)%\|", line)
        if pct:
            pbar_pct = f"{pct.group(1)}%"
        sp = __import__('re').search(r"([\d.]+)(?:it/s|s/it)", line)
        if sp:
            val = float(sp.group(1))
            if sp.group(0).endswith("s/it"):
                speed = f"{val:.2f}s/it"
            else:
                speed = f"{val:.2f}it/s"
        ls = __import__('re').search(r"loss=([\d.]+)", line)
        if ls:
            loss = ls.group(1)
        lr_m = __import__('re').search(r"lr=([\de.\-]+)", line)
        if lr_m:
            lr = f"{float(lr_m.group(1)):.2e}"
        st = __import__('re').search(r"step=(\d+)", line)
        if st:
            step = st.group(1)

    return {
        "epoch": epoch,
        "progress": pbar_pct,
        "speed": speed,
        "loss": loss,
        "lr": lr,
        "step": step,
        "running": process_running,
        "raw_progress": tqdm_out if tqdm_out != "NO_TQDM" else "—",
    }


def print_status(gpus, progress, log_name="erp3"):
    """Print formatted status."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_icon = "[RUN]" if progress["running"] else "[STOP]"

    print(f"\n{'='*60}")
    print(f"  {log_name.upper()} 训练监控  |  {now}  |  {status_icon} {'运行中' if progress['running'] else '已停止'}")
    print(f"{'='*60}")

    # GPU
    for g in gpus:
        mem_pct = int(g["mem_used"]) / int(g["mem_total"]) * 100 if int(g["mem_total"]) > 0 else 0
        bar = "█" * int(mem_pct // 5) + "░" * (20 - int(mem_pct // 5))
        print(f"  GPU {g['index']}: {g['name']}  |  util={g['util']}%  |  "
              f"mem={g['mem_used']}/{g['mem_total']} MiB  |  {g['temp']}°C")
        print(f"  [{bar}] {mem_pct:.0f}%")

    # Training
    print(f"\n  ── 训练进度 ──")
    print(f"     Epoch   : {progress['epoch']}")
    print(f"     进度    : {progress['progress']}  (step {progress['step']})")
    print(f"     Loss    : {progress['loss']}")
    print(f"     LR      : {progress['lr']}")
    print(f"     速度    : {progress['speed']}")
    if progress["raw_progress"] and progress["raw_progress"] != "—":
        # Show a cleaner version of the pbar
        raw = progress["raw_progress"]
        if len(raw) > 100:
            raw = raw[:100] + "..."
        print(f"     pbar    : {raw}")


def main():
    parser = argparse.ArgumentParser(description="点心杯远程训练监控")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="刷新间隔（秒，默认 2.0）")
    parser.add_argument("--log", "-l", type=str, default="erp4", help="监控的日志名（erp1/erp3/erp4，默认 erp4）")
    parser.add_argument("--once", "-o", action="store_true", help="单次输出，不循环")
    args = parser.parse_args()

    # Check paramiko
    try:
        import paramiko
    except ImportError:
        print("[ERROR] 需要 paramiko：pip install paramiko")
        sys.exit(1)

    print(f"连接 {HOST}:{PORT} ...")
    try:
        client = get_client()
    except Exception as e:
        print(f"[ERROR] 连接失败: {e}")
        sys.exit(1)

    try:
        if args.once:
            gpus = get_gpu_info(client)
            progress = get_training_progress(client, args.log)
            print_status(gpus, progress, args.log)
        else:
            print(f"按 Ctrl+C 停止监控（刷新间隔 {args.interval}s）")
            while True:
                try:
                    gpus = get_gpu_info(client)
                    progress = get_training_progress(client, args.log)
                    print_status(gpus, progress, args.log)
                    time.sleep(args.interval)
                except KeyboardInterrupt:
                    print("\n监控已停止。")
                    break
                except Exception as e:
                    print(f"\n⚠️ 读取异常: {e}，{args.interval}s 后重试...")
                    time.sleep(args.interval)
    finally:
        client.close()


if __name__ == "__main__":
    main()
