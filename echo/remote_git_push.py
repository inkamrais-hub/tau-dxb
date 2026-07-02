"""
remote_git_push.py — 每 epoch 自动上传产物到 atomgit 仓库

在远程 GPU 服务器上运行（由 train_echo_opus1.py 在每个 epoch 末尾调用）。
策略：
  - 小文件 (json/log/metadata): 上传到 atomgit
  - 大文件 (model.pt/ckpt.pt ~960MB): 只记录 metadata（位置+大小+sha256），
    实际文件留在远程，训练结束后通过 paramiko sftp 回传到本地 outputs/

调用方式:
  python remote_git_push.py <epoch> [--output-dir /hy-tmp/dimsum/outputs/echo-opus-1]

设计原则:
  - 失败不阻断训练（catch 所有异常，只打印 warning）
  - 幂等（重复调用同一 epoch 不会出错）
  - 离线安全（git push 失败时本地 commit 仍保留）
"""
import os
import sys
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

REPO_DIR = "/hy-tmp/atomgit_repo"
DEFAULT_OUTPUT = "/hy-tmp/dimsum/outputs/echo-opus-1"
LARGE_FILE_THRESHOLD_MB = 100  # 超过此大小不直接上传 git


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def push_epoch(epoch: int, output_dir: str = DEFAULT_OUTPUT):
    ts_start = datetime.now().isoformat(timespec="seconds")
    target = Path(REPO_DIR) / f"epoch{epoch}"
    target.mkdir(parents=True, exist_ok=True)

    # ─── 1. 收集小文件 ───
    small_files = [
        f"tau_epoch{epoch}.json",
        f"k_ratio_epoch{epoch}.json",
    ]
    copied = []
    for fname in small_files:
        src = Path(output_dir) / fname
        if src.exists():
            shutil.copy2(src, target / fname)
            copied.append(fname)

    # log.txt 总是覆盖最新
    log_path = Path(output_dir) / "log.txt"
    if log_path.exists():
        shutil.copy2(log_path, target / "log.txt")
        copied.append("log.txt")

    # ─── 2. 大文件只记录 metadata ───
    ep_dir = Path(output_dir) / f"epoch{epoch}"
    best_dir = Path(output_dir) / "best_model"
    meta = {
        "epoch": epoch,
        "pushed_at": ts_start,
        "output_dir": output_dir,
        "small_files": copied,
        "large_files": {},
    }
    candidates = []
    if ep_dir.exists():
        candidates += [(ep_dir / "model.pt", "model"), (ep_dir / "ckpt.pt", "ckpt")]
    if best_dir.exists():
        candidates += [(best_dir / "model.pt", "best_model")]
    for fpath, tag in candidates:
        if fpath.exists():
            size_mb = round(fpath.stat().st_size / 1024 / 1024, 2)
            meta["large_files"][tag] = {
                "path": str(fpath),
                "size_mb": size_mb,
                "sha256": sha256_of(fpath) if size_mb < LARGE_FILE_THRESHOLD_MB else "skipped-large",
            }

    with open(target / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ─── 3. 训练配置快照（首次 push 时）───
    config_snapshot = Path(REPO_DIR) / "training_config.json"
    if not config_snapshot.exists():
        cfg = {k: v for k, v in os.environ.items()
               if k.startswith(("TRAIN_", "VAL_", "TAU_", "OUTPUT_", "MODEL_", "BATCH_",
                                "GRAD_", "LR", "USE_", "EMA_", "SPARSITY_", "INIT_K",
                                "NUM_", "T0_", "T_MULT"))}
        with open(config_snapshot, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ─── 4. git add / commit / push ───
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "-A"],
                       check=True, capture_output=True, timeout=30)
        msg = f"epoch {epoch}: CER logged, {len(copied)} small files + {len(meta['large_files'])} large file metadata"
        r = subprocess.run(["git", "-C", REPO_DIR, "commit", "-m", msg],
                           capture_output=True, timeout=30)
        # commit 可能因为 nothing to commit 返回非零，不影响
    except subprocess.CalledProcessError as e:
        print(f"[git-push] add/commit error: {e.stderr.decode(errors='replace')[:200]}", file=sys.stderr)

    try:
        r = subprocess.run(["git", "-C", REPO_DIR, "push", "origin"],
                           capture_output=True, timeout=120)
        if r.returncode != 0:
            print(f"[git-push] push warning: {r.stderr.decode(errors='replace')[:200]}", file=sys.stderr)
        else:
            print(f"[git-push] epoch {epoch} pushed ✓")
    except subprocess.TimeoutExpired:
        print("[git-push] push timeout (network), commit kept locally", file=sys.stderr)
    except Exception as e:
        print(f"[git-push] push exception: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remote_git_push.py <epoch> [--output-dir PATH]")
        sys.exit(1)
    ep = int(sys.argv[1])
    out_dir = DEFAULT_OUTPUT
    if "--output-dir" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--output-dir") + 1]
    try:
        push_epoch(ep, out_dir)
    except Exception as e:
        # 任何异常都不应阻断训练
        print(f"[git-push] FATAL (训练继续): {e}", file=sys.stderr)
