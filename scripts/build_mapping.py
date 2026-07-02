"""
Build complete training dataset mapping.
- Extract 5-digit number from original audio filename
- Map to index.csv by 序号
- Produce audio_path → text pairs ready for training
"""
import csv, re, os, json, shutil

AUDIO_DIR = "/root/dimsum/data/audio_data"
INDEX_CSV = "/root/dimsum/data/index.csv"
MAPPING_CSV = f"{AUDIO_DIR}/mapping.csv"
OUTPUT_DIR = "/root/dimsum/data/prepared"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load index
index = {}
with open(INDEX_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row["序号"])
        index[idx] = {
            "yue": row["粤语原文"],
            "cmn": row["普通话翻译"],
            "jyutping": row["香港语言学会粤拼"],
            "scene": row["场景"],
        }

# Load mapping and build audio→text dataset
dataset = []
unmatched = []
seen_nums = set()

with open(MAPPING_CSV, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("short_name") or "__MACOSX" in line:
            continue
        short_name, orig_path = line.split(",", 1)

        # Skip .DS_Store
        if orig_path.endswith(".DS_Store"):
            continue

        # Extract 5-digit number from original filename
        fname = os.path.basename(orig_path)
        m = re.search(r'(\d{4,5})', fname)
        if not m:
            unmatched.append((short_name, fname))
            continue

        seq_num = int(m.group(1))
        if seq_num not in index:
            unmatched.append((short_name, fname))
            continue

        # Map audio file to text
        audio_path = os.path.join(AUDIO_DIR, short_name)
        audio_size = os.path.getsize(audio_path)

        dataset.append({
            "audio_path": audio_path,
            "audio_size": audio_size,
            "seq_num": seq_num,
            "text": index[seq_num]["yue"],
            "scene": index[seq_num]["scene"],
        })
        seen_nums.add(seq_num)

print(f"Total matched: {len(dataset)}")
print(f"Unmatched: {len(unmatched)}")

# Check how many index entries we covered
covered = len(seen_nums)
total_index = len(index)
print(f"Unique index entries covered: {covered}/{total_index}")

# Save as JSONL for easy training loading
with open(f"{OUTPUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Training data saved: {OUTPUT_DIR}/train.jsonl")

# Split into train/val (95/5)
import random
random.seed(42)
random.shuffle(dataset)
split = int(len(dataset) * 0.95)
train_set = dataset[:split]
val_set = dataset[split:]

with open(f"{OUTPUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
    for item in train_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open(f"{OUTPUT_DIR}/val.jsonl", "w", encoding="utf-8") as f:
    for item in val_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Train: {len(train_set)}, Val: {len(val_set)}")

# Also prepare test data mapping
# The test.jsonl has paths like ...4856-7500/04856....wav
# These extract by the same 5-digit number pattern
print("\n--- Test data mapping ---")
test_entries = []
with open("/root/dimsum/data/test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        test_entries.append(json.loads(line))

test_matched = 0
test_unmatched = 0
for entry in test_entries:
    ap = entry.get("audio_path", "")
    fname = os.path.basename(ap)
    m = re.search(r'(\d{4,5})', fname)
    if m:
        test_matched += 1
    else:
        test_unmatched += 1
        print(f"  Unmatched test: {fname[:50]}")

print(f"Test entries: {len(test_entries)}, can extract ID: {test_matched}, cannot: {test_unmatched}")
