"""
erp3 远程训练监控脚本
用法: python scripts/monitor_erp3.py
"""
import paramiko, time, os, sys, json
from datetime import datetime

HOST = "i-1.gpushare.com"
PORT = 59010
USER = "root"
PASSWORD = "HMAV6TEcCARYegYCFEGB6B89sDqqSGfU"
LOG_PATH = "/hy-tmp/dimsum/erp3.log"

# Refresh every 0.5s if update detected, otherwise 2s
FAST_POLL = 0.5
SLOW_POLL = 2.0
LONG_PAUSE = 60  # how often to show full status when no update

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)

last_size = 0
last_full_status = 0
step, epoch, loss, speed, eta = "?", "?", "?", "?", "?"
last_progress_line = ""

def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

def poll():
    global last_size, last_full_status, step, epoch, loss, speed, eta, last_progress_line
    
    # Check log size to detect changes quickly
    stdin, stdout, stderr = client.exec_command(f"stat --format=%s {LOG_PATH} 2>/dev/null || echo 0")
    size = int(stdout.read().decode().strip())
    
    if size > last_size:
        # Get last progress line
        stdin, stdout, stderr = client.exec_command(f"tail -1 {LOG_PATH}")
        line = stdout.read().decode().strip()
        if line and line != last_progress_line:
            last_progress_line = line
            # Parse: Epoch X/Y:  N%|...| N/T [MM:SS<HH:MM, X.XXs/it, loss=..., lr=..., step=N]
            parts = line.split()
            for p in parts:
                if p.startswith("Epoch"):
                    epoch = p.split("/")[0].replace("Epoch","")
                if "loss=" in p:
                    loss = p.replace("loss=","").rstrip(",")
                if "s/it" in p:
                    speed = p.rstrip(",")
                if "<" in p:
                    eta_part = p.split("<")[-1].rstrip(",")
                    if ":" in eta_part and not eta_part.startswith("s/it"):
                        eta = eta_part
            # Step progress
            if "/" in line:
                try:
                    prog = line.split("|")[1].strip().split("/")[0] if "|" in line else ""
                    if prog.isdigit():
                        step = int(prog)
                except: pass
        
        last_size = size
        return True
    
    # Periodic full status
    now = time.time()
    if now - last_full_status > LONG_PAUSE:
        return True
        
    return False

def show_full():
    global last_full_status
    last_full_status = time.time()
    
    stdin, stdout, stderr = client.exec_command("tail -8 /hy-tmp/dimsum/erp3.log")
    log_tail = stdout.read().decode().strip()
    
    stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits")
    gpu_raw = stdout.read().decode().strip().split(",")
    
    gpu_mem_used = gpu_raw[0].strip() if len(gpu_raw) > 0 else "?"
    gpu_mem_free = gpu_raw[1].strip() if len(gpu_raw) > 1 else "?"
    gpu_util = gpu_raw[2].strip() if len(gpu_raw) > 2 else "?"
    gpu_temp = gpu_raw[3].strip() if len(gpu_raw) > 3 else "?"
    
    # Epoch progress from log
    stdin, stdout, stderr = client.exec_command("grep -c 'Epoch 1/5' /hy-tmp/dimsum/erp3.log")
    epoch_done = 0
    try:
        epoch_done = int(stdout.read().decode().strip())
    except: pass
    
    print(f"\n{'='*60}")
    print(f"[{fmt_time(time.time())}] erp3 训练监控")
    print(f"  epoch: {epoch_done+1}/5" if epoch_done < 5 else "  epoch: 5/5 ✅")
    print(f"  步数: {step}/2113 | loss: {loss} | 速度: {speed} | ETA: {eta}")
    print(f"  GPU: {gpu_mem_used}/{gpu_mem_free} MB | 利用率: {gpu_util}% | 温度: {gpu_temp}°C")
    print(f"  {log_tail.split(chr(10))[-1] if log_tail else ''}")

print(f"[{fmt_time(time.time())}] erp3 监控启动 (0.5s 快速轮询)")
print(f"  按 Ctrl+C 终止\n")

try:
    while True:
        changed = poll()
        if changed:
            show_full()
        
        # Fast poll (0.5s) if actively training, slow (2s) if idle
        stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0")
        util = stdout.read().decode().strip()
        try:
            util_int = int(util)
            delay = FAST_POLL if util_int > 0 else SLOW_POLL
        except:
            delay = SLOW_POLL
        
        time.sleep(delay)

except KeyboardInterrupt:
    print(f"\n[{fmt_time(time.time())}] 监控已终止")
finally:
    client.close()
