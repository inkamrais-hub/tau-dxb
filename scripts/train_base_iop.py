"""
Train base-iop: a fast-cycled + slow-annealed decoder-only fine-tune of whisper-small.
Used as the base model for full τ* estimation (encoder/decoder/cross).
"""
import os, json, time
os.environ["HF_HOME"] = "/hf_cache"
os.environ["TORCH_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
from tqdm import tqdm
import jiwer

OUTPUT_DIR = "/root/dimsum/outputs/base_iop"
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 1e-4
NUM_EPOCHS = 2
MAX_LENGTH = 128
SAMPLING_RATE = 16000
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
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        if audio.ndim > 1:
            audio = audio.mean(-1)
        inp = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        )
        lbl = self.processor.tokenizer(
            item["text"], truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        return {
            "input_features": inp.input_features.squeeze(0),
            "labels": lbl.input_ids.squeeze(0),
            "text": item["text"],
            "audio_path": item["audio_path"],
        }


def collate_fn(batch):
    max_feat = max(b["input_features"].shape[-1] for b in batch)
    fdim = batch[0]["input_features"].shape[-2]
    inp = torch.zeros(len(batch), fdim, max_feat)
    for i, b in enumerate(batch):
        inp[i, :, :b["input_features"].shape[-1]] = b["input_features"]
    max_lbl = max(b["labels"].shape[-1] for b in batch)
    lbl = torch.full((len(batch), max_lbl), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        lbl[i, :b["labels"].shape[-1]] = b["labels"]
    return {
        "input_features": inp,
        "labels": lbl,
        "text": [b["text"] for b in batch],
        "audio_path": [b["audio_path"] for b in batch],
    }


def evaluate_cer(model, processor, val_loader):
    model.eval()
    refs, hyps = [], []
    gen_kwargs = {
        "language": "zh",
        "task": "transcribe",
        "num_beams": 1,
        "temperature": 0.0,
        "do_sample": False,
    }
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  eval CER"):
            inputs = batch["input_features"].to(DEVICE)
            preds = model.generate(inputs, **gen_kwargs)
            text = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
            refs.extend(batch["text"])
            hyps.extend(text)
    model.train()
    return jiwer.cer(refs, hyps)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[base-iop] Device: {DEVICE}")
    print(f"[base-iop] epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LR}")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", cache_dir="/hf_cache",
        language="zh", task="transcribe"
    )
    train_ds = CantoneseASRDataset("/root/dimsum/data/prepared/train.jsonl", processor)
    val_ds = CantoneseASRDataset("/root/dimsum/data/prepared/val.jsonl", processor)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", attn_implementation="sdpa", cache_dir="/hf_cache"
    )
    # Freeze encoder, train decoder only (fast + stable for whisper adaptation)
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    model.to(DEVICE)

    optimizer = AdamW(model.model.decoder.parameters(), lr=LR, weight_decay=0.01)
    # Fast cycles early, slower cycles later: T_0=200, T_mult=2
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)

    global_step, best_cer = 0, float("inf")
    log_lines = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for bidx, batch in enumerate(pbar):
            loss = model(
                input_features=batch["input_features"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            ).loss / GRAD_ACCUM
            loss.backward()

            if (bidx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.model.decoder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * GRAD_ACCUM
            n_steps += 1
            if global_step % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item() * GRAD_ACCUM:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })

        avg_loss = total_loss / n_steps
        cer = evaluate_cer(model, processor, val_loader)
        log_line = f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, val_CER={cer*100:.2f}%, lr={scheduler.get_last_lr()[0]:.2e}, steps={global_step}"
        print(f"\n{log_line}")
        log_lines.append(log_line)

        # save per-epoch
        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))

        if cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            print(f"  New best CER: {best_cer*100:.2f}% -> saved to {best_dir}")

    # final
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))

    with open(os.path.join(OUTPUT_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n[base-iop] Done. Best val CER: {best_cer*100:.2f}%")


if __name__ == "__main__":
    main()
