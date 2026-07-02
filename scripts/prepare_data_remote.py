"""
Download and prepare the DimSum Cantonese ASR dataset on the remote GPU machine.
Assumes workspace at /hy-tmp/dimsum and HF cache at /hy-tmp/hf_cache.
"""
import os
import sys
import json
import csv
import re
import shutil
import zipfile
import subprocess

os.environ["HF_HOME"] = "/hy-tmp/hf_cache"
os.environ["TORCH_HOME"] = "/hy-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

DATA_DIR = "/hy-tmp/dimsum/data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_data")
PREPARED_DIR = os.path.join(DATA_DIR, "prepared")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(PREPARED_DIR, exist_ok=True)


def ensure_hf_hub():
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
        return hf_hub_download


def download_repo_file(hf_hub_download, repo_id, filename, local_dir):
    print(f"Downloading {filename} from {repo_id} ...")
    local_path = os.path.join(local_dir, filename)

    # Use aria2c multi-connection download for large files
    if filename == "data.zip" and shutil.which("aria2c"):
        url = f"{os.environ['HF_ENDPOINT']}/datasets/{repo_id}/resolve/main/{filename}"
        print(f"  Using aria2c multi-connection download from {url}")
        if os.path.exists(local_path):
            print(f"  Resuming existing file: {local_path}")
        subprocess.check_call([
            "aria2c", "-s", "16", "-x", "16", "-k", "1M",
            "--continue=true", "--file-allocation=none",
            "-d", local_dir, "-o", filename, url,
        ])
        print(f"  -> {local_path}")
        return local_path

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir)
    print(f"  -> {path}")
    return path


def unzip_with_short_names(zip_path, dst):
    print(f"Extracting {zip_path} to {dst} with short names ...")
    mapping = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        infos = [info for info in z.infolist()
                 if not info.is_dir() and '__MACOSX' not in info.filename and not info.filename.startswith('/')]
        for idx, info in enumerate(infos):
            ext = os.path.splitext(info.filename)[1] or '.wav'
            short_name = f"audio_{idx:05d}{ext}"
            out_path = os.path.join(dst, short_name)
            with z.open(info) as srcf, open(out_path, 'wb') as dstf:
                shutil.copyfileobj(srcf, dstf)
            mapping.append((short_name, info.filename))
    mapping_path = os.path.join(dst, 'mapping.csv')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.write('short_name,original_path\n')
        for s, o in mapping:
            f.write(f'{s},{o}\n')
    print(f"Extracted {len(mapping)} files. Mapping saved to {mapping_path}")


def build_prepared_jsonl(index_csv, mapping_csv, audio_dir, output_dir):
    print("Building prepared train/val JSONL ...")
    index = {}
    with open(index_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["序号"])
            except Exception:
                continue
            index[idx] = {
                "yue": row.get("粤语原文", ""),
                "cmn": row.get("普通话翻译", ""),
                "jyutping": row.get("香港语言学会粤拼", ""),
                "scene": row.get("场景", ""),
            }

    dataset = []
    with open(mapping_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("short_name") or "__MACOSX" in line:
                continue
            short_name, orig_path = line.split(",", 1)
            if orig_path.endswith(".DS_Store"):
                continue
            fname = os.path.basename(orig_path)
            m = re.search(r'(\d{4,5})', fname)
            if not m:
                continue
            seq_num = int(m.group(1))
            if seq_num not in index:
                continue
            audio_path = os.path.join(audio_dir, short_name)
            if not os.path.exists(audio_path):
                continue
            dataset.append({
                "audio_path": audio_path,
                "audio_size": os.path.getsize(audio_path),
                "seq_num": seq_num,
                "text": index[seq_num]["yue"],
                "scene": index[seq_num]["scene"],
            })

    print(f"Total matched: {len(dataset)} / {len(index)} index entries")

    import random
    random.seed(42)
    random.shuffle(dataset)
    split = int(len(dataset) * 0.95)
    train_set = dataset[:split]
    val_set = dataset[split:]

    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")


def main():
    hf_hub_download = ensure_hf_hub()
    repo_id = "leeduckgo/cantonese-life-scenarios-corpus"

    # Download required files
    zip_path = download_repo_file(hf_hub_download, repo_id, "data.zip", DATA_DIR)
    index_path = download_repo_file(hf_hub_download, repo_id, "index.csv", DATA_DIR)
    test_path = download_repo_file(hf_hub_download, repo_id, "test.jsonl", DATA_DIR)
    eval_path = download_repo_file(hf_hub_download, repo_id, "evaluator.py", DATA_DIR)
    sec_eval_path = download_repo_file(hf_hub_download, repo_id, "sec_evaluator.py", DATA_DIR)

    # Extract audio
    unzip_with_short_names(zip_path, AUDIO_DIR)

    # Build train/val JSONL
    mapping_path = os.path.join(AUDIO_DIR, "mapping.csv")
    build_prepared_jsonl(index_path, mapping_path, AUDIO_DIR, PREPARED_DIR)

    print("\nDataset preparation complete.")
    print(f"  Prepared: {PREPARED_DIR}")
    print(f"  Audio:    {AUDIO_DIR}")


if __name__ == "__main__":
    main()
