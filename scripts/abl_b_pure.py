"""
Ablation B: Pure decoder fine-tune (no τ-opus, no SparseGrad).
"""
import os, json
os.environ["HF_HOME"] = "/hf_cache"
os.environ["TORCH_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
from tqdm import tqdm

OUTPUT_DIR = "/root/dimsum/outputs/abl_b_pure"
BATCH_SIZE = 4; GRAD_ACCUM = 4; LR = 5e-5; NUM_EPOCHS = 5
MAX_LENGTH = 128; SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CantoneseASRDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.processor = processor
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        audio, sr = sf.read(item["audio_path"])
        if sr != SAMPLING_RATE:
            import librosa; audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        if audio.ndim > 1: audio = audio.mean(-1)
        inp = self.processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        lbl = self.processor.tokenizer(item["text"], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {"input_features": inp.input_features.squeeze(0), "labels": lbl.input_ids.squeeze(0)}

def collate_fn(batch):
    max_feat = max(b["input_features"].shape[-1] for b in batch)
    fdim = batch[0]["input_features"].shape[-2]
    inp = torch.zeros(len(batch), fdim, max_feat)
    for i, b in enumerate(batch): inp[i, :, :b["input_features"].shape[-1]] = b["input_features"]
    max_lbl = max(b["labels"].shape[-1] for b in batch)
    lbl = torch.full((len(batch), max_lbl), -100, dtype=torch.long)
    for i, b in enumerate(batch): lbl[i, :b["labels"].shape[-1]] = b["labels"]
    return {"input_features": inp, "labels": lbl}

def main():
    print(f"[Abl B] Pure finetune | Device: {DEVICE}")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir="/hf_cache", language="zh", task="transcribe")
    train_ds = CantoneseASRDataset("/root/dimsum/data/prepared/train.jsonl", processor)
    val_ds = CantoneseASRDataset("/root/dimsum/data/prepared/val.jsonl", processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", attn_implementation="sdpa", cache_dir="/hf_cache")
    for p in model.model.encoder.parameters(): p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    model.to(DEVICE)

    optimizer = AdamW(model.model.decoder.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) * NUM_EPOCHS) // GRAD_ACCUM
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    global_step, best_loss = 0, float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train(); total_loss = 0.0; n_steps = 0; optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for bidx, batch in enumerate(pbar):
            loss = model(input_features=batch["input_features"].to(DEVICE), labels=batch["labels"].to(DEVICE)).loss / GRAD_ACCUM
            loss.backward()
            if (bidx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.model.decoder.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); global_step += 1
            total_loss += loss.item() * GRAD_ACCUM; n_steps += 1
            if global_step % 10 == 0: pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM:.4f}"})
        avg_loss = total_loss / n_steps
        print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.join(OUTPUT_DIR, "best_model"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model", "model.pt"))
    os.makedirs(os.path.join(OUTPUT_DIR, "final_model"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model", "model.pt"))
    print("[Abl B] Done!")

if __name__ == "__main__": main()
