"""Check data leakage between train/val/test."""
import json


def load_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                ap = d.get("audio_path", "")
                utt = d.get("utt_id", d.get("id", ""))
                txt = d.get("text", "")
                ids.add((utt, ap, txt))
    return ids


TRAIN_PATH = "/root/dimsum/data/prepared/train.jsonl"
VAL_PATH = "/root/dimsum/data/prepared/val.jsonl"
TEST_PATH = "/root/dimsum/data/test.jsonl"

train = load_ids(TRAIN_PATH)
val = load_ids(VAL_PATH)
test = load_ids(TEST_PATH)

train_aps = {x[1] for x in train}
val_aps = {x[1] for x in val}
test_aps = {x[1] for x in test}

tv = train_aps & test_aps
vv = val_aps & test_aps

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Train-Test audio_path overlap: {len(tv)} samples")
print(f"Val-Test audio_path overlap: {len(vv)} samples")

if tv:
    for t in train:
        if t[1] in tv:
            print(f"  OVERLAP: utt={t[0]}, path={t[1]}, text={t[2][:50]}")
            break

# Check text overlap too
train_texts = {x[2] for x in train}
test_texts = {x[2] for x in test}
val_texts = {x[2] for x in val}
tv_text = train_texts & test_texts
vv_text = val_texts & test_texts
print(f"Train-Test text overlap: {len(tv_text)}")
print(f"Val-Test text overlap: {len(vv_text)}")

# Check if test.jsonl is all from val set
all_train_val = train_aps | val_aps
test_in_trainval = test_aps & all_train_val
print(f"Test samples in train+val: {len(test_in_trainval)}")
if test_in_trainval:
    print("WARNING: test leakage detected!")
else:
    print("OK: no audio_path leakage between test and train/val")

# Also check batch_id in test.jsonl
print("\nSample test.jsonl entries:")
with open(TEST_PATH) as f:
    for i, line in enumerate(f):
        if i < 3:
            d = json.loads(line)
            print(f"  {d.get('utt_id', '?')} | {d.get('audio_path', '?')} | {d.get('text', '?')[:40]}")
