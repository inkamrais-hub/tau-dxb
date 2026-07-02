"""Check data mapping between index.csv and audio files."""
import csv
import os

# read index.csv to get text labels
with open("/root/dimsum/data/index.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    index_rows = []
    for row in reader:
        index_rows.append(row)

# read mapping.csv (short_name, original_path)
mapping = []
with open("/root/dimsum/data/audio_data/mapping.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("short_name"):
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            mapping.append((parts[0], parts[1]))

print(f"Index rows: {len(index_rows)}")
print(f"Audio files: {len(mapping)}")

print("\nFirst 5 index rows:")
for r in index_rows[:5]:
    print(f"  #{r.get('序号','')} yue={r.get('粤语原文','')[:30]} cmn={r.get('普通话翻译','')[:20]}")

print("\nFirst 5 mapping (short -> original):")
for short, orig in mapping[:5]:
    print(f"  {short} -> {os.path.basename(orig)}")

# The mapping from original filename pattern to index number
# Original filenames contain 4-5 digit number prefix like 00001, 00025, etc.
# This should match 序号 in index.csv
print("\n--- Checking if audio filenames contain index numbers ---")
import re
sample_count = 0
for short, orig in mapping[:20]:
    fname = os.path.basename(orig)
    match = re.search(r'(\d{4,5})', fname)
    if match:
        num = match.group(1)
        print(f"  {short}: extracted number {num} from {fname[:40]}")
        sample_count += 1

print(f"\n  {sample_count} samples have extractable number prefixes")

# Also check the audio_data folder for the wav count
import glob
wav_files = glob.glob("/root/dimsum/data/audio_data/audio_*.wav")
print(f"\nTotal WAV files: {len(wav_files)}")

# Some stats on file sizes
import os
sizes = [os.path.getsize(f) for f in wav_files[:10]]
print(f"First 10 file sizes: {sizes}")
