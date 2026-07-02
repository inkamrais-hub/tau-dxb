"""
Paramiko-based remote management for the DimSum GPU server.
Replaces brittle shell SSH with a persistent Python SSH client.

Usage:
  python remote.py upload          -- upload local scripts to remote
  python remote.py start-erp1      -- kill old training, upload, start erp1
  python remote.py status          -- check training status + GPU
  python remote.py logs [N]        -- tail last N lines of erp1 log
  python remote.py exec <cmd>      -- run arbitrary command
  python remote.py kill            -- kill training processes
"""
import os
import sys
import time
import argparse
from pathlib import Path

HOST = "i-1.gpushare.com"
PORT = 59010
USER = "root"
PASSWORD = "HMAV6TEcCARYegYCFEGB6B89sDqqSGfU"

REMOTE_BASE = "/hy-tmp/dimsum"
LOCAL_SCRIPTS = Path(__file__).resolve().parent

FILES_TO_UPLOAD = [
    "patch_whisper_sparse_grad.py",
    "patch_whisper_stau.py",
    "stau_opus_operator.py",
    "triton_wta.py",
    "compute_tau_star_echo.py",
    "train_erp1.py",
    "train_erp3.py",
    "train_erp4.py",
    "train_echo_opus1.py",
    "_smoke_erp3_v2.py",
    "tau_star_solver.py",
    "compute_tau_star_iop.py",
    "prepare_data_remote.py",
    "setup_data_5090.py",
    "launch_echo_opus1.sh",
    "remote_git_push.py",
]

# External files (not in scripts/) needed on the server
EXTRA_UPLOADS = {
    "tau_star_opus.py": str(Path(__file__).resolve().parent.parent.parent / "τopus" / "tau_star_opus.py"),
}

# τ* JSON for echo-opus-1
TAU_STAR_LOCAL = Path(__file__).resolve().parent.parent / "outputs" / "echo-opus-1" / "tau_star.json"
TAU_STAR_REMOTE = f"{REMOTE_BASE}/outputs/echo-opus-1/tau_star.json"


def get_client():
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    return client


def upload_files(client, files=None):
    """Upload script files to remote server."""
    if files is None:
        files = FILES_TO_UPLOAD
    sftp = client.open_sftp()
    remote_dir = f"{REMOTE_BASE}/scripts"
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        client.exec_command(f"mkdir -p {remote_dir}")

    for fname in files:
        local = LOCAL_SCRIPTS / fname
        if not local.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        remote = f"{remote_dir}/{fname}"
        sftp.put(str(local), remote)
        print(f"  OK {fname} -> {remote}")

    # Upload external files too
    for fname, local_path in EXTRA_UPLOADS.items():
        local = Path(local_path)
        if not local.exists():
            print(f"  SKIP {fname} (not found at {local_path})")
            continue
        remote = f"{remote_dir}/{fname}"
        sftp.put(str(local), remote)
        print(f"  OK {fname} -> {remote}")

    sftp.close()


def run_cmd(client, cmd, stream=False, timeout=120):
    """Run a command and return stdout. If stream=True, stream in realtime."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    if stream:
        for line in iter(stdout.readline, ""):
            print(line, end="", flush=True)
        exit_code = stdout.channel.recv_exit_status()
        return stdout, stderr
    else:
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return out, err


def cmd_status(client):
    """Check training status and GPU."""
    ps_out, _ = run_cmd(client, "ps aux | grep -E 'train_(erp1|echo_opus1)' | grep -v grep || echo 'NOT RUNNING'")
    gpu_out, _ = run_cmd(client, "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
    log_out, _ = run_cmd(client, "(tail -15 /hy-tmp/dimsum/echo-opus-1.log 2>/dev/null || tail -15 /hy-tmp/dimsum/erp1.log 2>/dev/null) || echo 'No log yet'")

    print("=== PROCESS ===")
    print(ps_out.strip())
    print("\n=== GPU ===")
    print(gpu_out.strip())
    print("\n=== LAST LOG ===")
    print(log_out.strip())


def cmd_logs(client, n=20, tag="echo-opus-1"):
    """Tail training log."""
    log_path = f"/hy-tmp/dimsum/{tag}.log" if tag else "/hy-tmp/dimsum/echo-opus-1.log"
    out, _ = run_cmd(client, f"tail -{n} {log_path} 2>/dev/null || echo 'No log yet'")
    print(out.strip())


def cmd_kill(client):
    """Kill training processes."""
    run_cmd(client, "pkill -f 'train_(erp1|echo_opus1)' 2>/dev/null; echo 'Killed'")
    time.sleep(1)


def cmd_exec(client, command):
    """Run arbitrary command."""
    out, err = run_cmd(client, command, timeout=300)
    if out:
        print(out.strip())
    if err:
        print(f"STDERR:\n{err.strip()}", file=sys.stderr)


def cmd_start_erp1(client):
    """Kill old training, upload files, start erp1."""
    print("1. Uploading scripts...")
    upload_files(client)

    print("\n2. Killing old processes...")
    cmd_kill(client)

    print("\n3. Checking data files...")
    out, _ = run_cmd(client, "wc -l /hy-tmp/dimsum/data/prepared/train.jsonl /hy-tmp/dimsum/data/prepared/val.jsonl 2>/dev/null || echo 'MISSING'")
    print(out.strip())

    print("\n4. Starting erp1 training...")
    run_script = f"""#!/bin/bash
export HF_HOME=/hy-tmp/hf_cache
export TORCH_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRAIN_JSONL=/hy-tmp/dimsum/data/prepared/train.jsonl
export VAL_JSONL=/hy-tmp/dimsum/data/prepared/val.jsonl
export OUTPUT_DIR=/hy-tmp/dimsum/outputs/base-erp1
export MODEL_NAME=/hy-tmp/whisper-small-local
export BATCH_SIZE=4
export GRAD_ACCUM=4
export LR=1e-4
export NUM_EPOCHS=2
export USE_BF16=1
export USE_EMA=1
export T_0=200
export T_MULT=2
python3 -u {REMOTE_BASE}/scripts/train_erp1.py >> /hy-tmp/dimsum/erp1.log 2>&1
"""
    # Write and execute
    run_cmd(client, f"cat > /tmp/start_erp1.sh << 'HEREDOC_EOF'\n{run_script}HEREDOC_EOF")
    run_cmd(client, "chmod +x /tmp/start_erp1.sh")

    print("  Launching nohup training...")
    stdin, stdout, stderr = client.exec_command(
        "nohup bash /tmp/start_erp1.sh &> /dev/null & echo PID=$!"
    )
    pid = stdout.read().decode().strip()
    print(f"  Started with PID={pid}")

    time.sleep(5)
    print("\n5. Status check:")
    cmd_status(client)


def cmd_download(client, remote_path, local_path):
    """Download a file from remote server via SFTP."""
    import os
    sftp = client.open_sftp()
    # Ensure local directory exists
    local_dir = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(local_dir, exist_ok=True)
    sftp.get(remote_path, local_path)
    sftp.close()
    print(f"Downloaded: {remote_path} -> {local_path}")


def upload_tau_star_json(client):
    """Upload tau_star.json to /hy-tmp/dimsum/model_zoo/erp3/ for τ init."""
    if not TAU_STAR_LOCAL.exists():
        print(f"  SKIP tau_star.json (not found at {TAU_STAR_LOCAL})")
        return
    sftp = client.open_sftp()
    remote_dir = str(Path(TAU_STAR_REMOTE).parent)
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        client.exec_command(f"mkdir -p {remote_dir}")
    sftp.put(str(TAU_STAR_LOCAL), TAU_STAR_REMOTE)
    sftp.close()
    print(f"  OK tau_star.json -> {TAU_STAR_REMOTE}")


def cmd_start_echo_opus1(client):
    """Step 1: τ* solver → Step 2: train echo-opus-1 (15 epochs)."""
    print("=" * 60)
    print("ECHO-OPUS-1: τ* solve → train pipeline")
    print("=" * 60)

    # ── Phase 1: Upload ──
    print("\n[Phase 1] Uploading scripts...")
    upload_files(client)

    # ── Phase 2: Kill old ──
    print("\n[Phase 2] Killing old processes...")
    cmd_kill(client)

    # ── Phase 3: Verify prerequisites ──
    print("\n[Phase 3] Verifying prerequisites...")
    out, _ = run_cmd(client, "wc -l /hy-tmp/dimsum/data/prepared/train.jsonl /hy-tmp/dimsum/data/prepared/val.jsonl 2>/dev/null || echo 'MISSING'")
    print(f"  Data: {out.strip()}")
    out2, _ = run_cmd(client, "ls /hy-tmp/whisper-small-local/model.safetensors 2>/dev/null || echo 'MISSING whisper-small'")
    print(f"  Base model: {out2.strip()}")

    # Check if τ-opus lib is synced (for TauStarOpus)
    out3, _ = run_cmd(client, "python3 -c 'from tau_star_opus import TauStarOpus; print(\"OK\")' 2>&1 || echo 'MISSING'")
    tau_opus_ok = "OK" in out3
    print(f"  tau_star_opus.py: {'OK' if tau_opus_ok else 'NEEDS SYNC (will try local upload)'}")

    # ── Phase 4: τ* solver (compute on server) ──
    print("\n[Phase 4] Computing τ* on server (per-type σ + KL validation)...")
    tau_solve_cmd = f"""#!/bin/bash
export HF_HOME=/hy-tmp/hf_cache
export TORCH_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /hy-tmp/dimsum/outputs/echo-opus-1
python3 -u {REMOTE_BASE}/scripts/compute_tau_star_echo.py \
  --data_jsonl /hy-tmp/dimsum/data/prepared/train.jsonl \
  --model /hy-tmp/whisper-small-local \
  --n_samples 20 \
  --output {TAU_STAR_REMOTE} \
  --max_audio_sec 8.0 2>&1
"""
    run_cmd(client, f"cat > /tmp/tau_solve.sh << 'HEREDOC_EOF'\n{tau_solve_cmd}HEREDOC_EOF")
    run_cmd(client, "chmod +x /tmp/tau_solve.sh")

    print("  Running τ* solver (this may take 2-5 minutes for 200 samples)...")
    out, err = run_cmd(client, "bash /tmp/tau_solve.sh", timeout=600)
    print(out[-3000:] if len(out) > 3000 else out)  # last 3000 chars
    if err and err.strip():
        print(f"  STDERR (last 500 chars): {err[-500:]}")

    # Verify τ* output
    out_verify, _ = run_cmd(client, f"python3 -c \"import json; d=json.load(open('{TAU_STAR_REMOTE}')); print(f'τ* OK: {{len(d)}} entries'); kl=[r['kl_mean'] for r in d]; print(f'KL mean: {{sum(kl)/len(kl):.6f}}, max: {{max(kl):.6f}}')\" 2>&1")
    print(f"  τ* verification: {out_verify.strip()}")

    # ── Phase 5: Launch training ──
    print("\n[Phase 5] Launching echo-opus-1 training (15 epochs)...")
    run_script = f"""#!/bin/bash
export HF_HOME=/hy-tmp/hf_cache
export TORCH_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRAIN_JSONL=/hy-tmp/dimsum/data/prepared/train.jsonl
export VAL_JSONL=/hy-tmp/dimsum/data/prepared/val.jsonl
export TAU_STAR_JSON={TAU_STAR_REMOTE}
export OUTPUT_DIR=/hy-tmp/dimsum/outputs/echo-opus-1
export MODEL_NAME=/hy-tmp/whisper-small-local
export BATCH_SIZE=4
export GRAD_ACCUM=4
export LR=5e-6
export TAU_LR=3e-4
export NUM_EPOCHS=15
export T0_EPOCHS=4
export T_MULT=2
export INIT_K_RATIO=0.4
export SPARSITY_LAMBDA=0.01
export USE_BF16=1
export USE_EMA=1
python3 -u {REMOTE_BASE}/scripts/train_echo_opus1.py >> /hy-tmp/dimsum/echo-opus-1.log 2>&1
"""
    run_cmd(client, f"cat > /tmp/start_echo_opus1.sh << 'HEREDOC_EOF'\n{run_script}HEREDOC_EOF")
    run_cmd(client, "chmod +x /tmp/start_echo_opus1.sh")

    print("  Launching nohup training...")
    stdin, stdout, stderr = client.exec_command(
        "nohup bash /tmp/start_echo_opus1.sh &> /dev/null & echo PID=$!"
    )
    pid = stdout.read().decode().strip()
    print(f"  Started with PID={pid}")

    time.sleep(5)
    print("\n[Phase 6] Status check:")
    cmd_status(client)


def cmd_start_erp4(client):
    """Kill old training, upload files, start erp4 (warm start from erp3)."""
    print("1. Uploading scripts...")
    upload_files(client)

    print("\n2. Killing old processes...")
    cmd_kill(client)

    print("\n3. Checking data files...")
    out, _ = run_cmd(client, "wc -l /hy-tmp/dimsum/data/prepared/train.jsonl /hy-tmp/dimsum/data/prepared/val.jsonl 2>/dev/null || echo 'MISSING'")
    print(out.strip())
    out2, _ = run_cmd(client, "ls -lh /hy-tmp/dimsum/outputs/erp3/epoch5/model.pt 2>/dev/null || echo 'MISSING ERP3 CKPT'")
    print(out2.strip())

    print("\n4. Starting erp4 training (7 epochs, warm start from erp3)...")
    run_script = f"""#!/bin/bash
export HF_HOME=/hy-tmp/hf_cache
export TORCH_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRAIN_JSONL=/hy-tmp/dimsum/data/prepared/train.jsonl
export VAL_JSONL=/hy-tmp/dimsum/data/prepared/val.jsonl
export ERP3_CKPT=/hy-tmp/dimsum/outputs/erp3/epoch5/model.pt
export TAU_STAR_JSON=/hy-tmp/dimsum/outputs/erp2_tau_star.json
export OUTPUT_DIR=/hy-tmp/dimsum/outputs/erp4
export MODEL_NAME=/hy-tmp/whisper-small-local
export BATCH_SIZE=4
export GRAD_ACCUM=4
export LR=5e-6
export TAU_LR=3e-4
export NUM_EPOCHS=7
export ENC_K_RATIO=0.4
export DEC_K_RATIO=0.3
export USE_BF16=1
export USE_EMA=1
python3 -u {REMOTE_BASE}/scripts/train_erp4.py >> /hy-tmp/dimsum/erp4.log 2>&1
"""
    run_cmd(client, f"cat > /tmp/start_erp4.sh << 'HEREDOC_EOF'\n{run_script}HEREDOC_EOF")
    run_cmd(client, "chmod +x /tmp/start_erp4.sh")

    print("  Launching nohup training...")
    stdin, stdout, stderr = client.exec_command(
        "nohup bash /tmp/start_erp4.sh &> /dev/null & echo PID=$!"
    )
    pid = stdout.read().decode().strip()
    print(f"  Started with PID={pid}")

    time.sleep(5)
    print("\n5. Status check:")
    cmd_status(client)


def main():
    parser = argparse.ArgumentParser(description="DimSum GPU remote manager")
    parser.add_argument("action", choices=["upload", "start-erp1", "start-erp4", "start-echo-opus-1", "status", "logs", "exec", "kill", "watch", "download"])
    parser.add_argument("args", nargs="*", help="Extra args (e.g. log lines, command)")
    args = parser.parse_args()

    print(f"Connecting to {HOST}:{PORT}...")
    try:
        client = get_client()
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure paramiko is installed: pip install paramiko")
        sys.exit(1)

    try:
        if args.action == "upload":
            upload_files(client)
        elif args.action == "start-erp1":
            cmd_start_erp1(client)
        elif args.action == "start-erp4":
            cmd_start_erp4(client)
        elif args.action == "start-echo-opus-1":
            cmd_start_echo_opus1(client)
        elif args.action == "status":
            cmd_status(client)
        elif args.action == "logs":
            n = int(args.args[0]) if args.args and args.args[0].isdigit() else 20
            tag = args.args[1] if args.args and len(args.args) > 1 else None
            cmd_logs(client, n, tag)
        elif args.action == "exec":
            cmd_exec(client, " ".join(args.args))
        elif args.action == "kill":
            cmd_kill(client)
        elif args.action == "download":
            if args.args:
                cmd_download(client, args.args[0], args.args[1] if len(args.args) > 1 else os.path.basename(args.args[0]))
            else:
                print("Usage: python remote.py download <remote_path> [local_path]")
        elif args.action == "watch":
            print("Monitoring training (Ctrl+C to stop)...")
            try:
                while True:
                    cmd_status(client)
                    print(f"\n{'='*60}\n")
                    time.sleep(30)
            except KeyboardInterrupt:
                print("\nStopped watching.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
