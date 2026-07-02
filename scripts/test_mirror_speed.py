"""Test download speed of data.zip from multiple HF mirrors."""
import os
import sys
import time
import subprocess

MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.sukunka.com",
    "https://hf.yzuu.cf",
    "https://hf.startup976.fun",
    "https://hf.wtsaigc.com",
    "https://hf-mirror.nju.edu.cn",
]

REPO = "leeduckgo/cantonese-life-scenarios-corpus"
FILE = "data.zip"
TEST_SECONDS = 15


def test_mirror(base_url):
    url = f"{base_url}/datasets/{REPO}/resolve/main/{FILE}"
    out_path = f"/tmp/test_{base_url.replace('https://', '').replace('/', '_')}.zip"
    cmd = [
        "curl", "-s", "-L", "-o", out_path,
        "--max-time", str(TEST_SECONDS + 5),
        url,
    ]
    start = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=TEST_SECONDS)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait()
    elapsed = time.time() - start
    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    speed_mbps = (size * 8 / elapsed / 1024 / 1024) if elapsed > 0 else 0
    speed_mbs = (size / elapsed / 1024 / 1024) if elapsed > 0 else 0
    print(f"{base_url:40s} | {size:>10d} bytes | {elapsed:>5.1f}s | {speed_mbps:>6.2f} Mbps | {speed_mbs:>6.2f} MB/s")
    try:
        os.remove(out_path)
    except Exception:
        pass
    return base_url, size, speed_mbps


if __name__ == "__main__":
    print(f"Testing mirrors for {REPO}/{FILE} ({TEST_SECONDS}s each)...")
    print("-" * 100)
    results = []
    for mirror in MIRRORS:
        try:
            results.append(test_mirror(mirror))
        except Exception as e:
            print(f"{mirror:40s} | ERROR: {e}")
    print("-" * 100)
    best = max(results, key=lambda x: x[2]) if results else None
    if best:
        print(f"Best mirror: {best[0]} ({best[2]:.2f} Mbps)")
