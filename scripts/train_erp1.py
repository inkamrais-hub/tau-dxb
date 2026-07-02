"""
Train base-erp1 (erp v2 stage 1).
Decoder-only whisper-small fine-tune on Cantonese life-scenarios corpus.
Config: bf16, τ-opus OFF, optional EMA.
"""
import os
import sys
import json

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/hy-tmp/hf_cache")
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/hy-tmp/hf_cache")
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
from tqdm import tqdm
import jiwer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from patch_whisper_sparse_grad import apply_sparse_grad_wta, set_sparse_grad_k_ratio

# Config
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", "/hy-tmp/dimsum/data/prepared/train.jsonl")
VAL_JSONL   = os.environ.get("VAL_JSONL", "/hy-tmp/dimsum/data/prepared/val.jsonl")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "/hy-tmp/dimsum/outputs/base-erp1")
MODEL_NAME  = os.environ.get("MODEL_NAME", "/hy-tmp/whisper-small-local")

BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))
LR           = float(os.environ.get("LR", "1e-4"))
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "2"))
MAX_LENGTH   = 128
SAMPLING_RATE = 16000
USE_BF16     = os.environ.get("USE_BF16", "1") == "1"
USE_EMA      = os.environ.get("USE_EMA", "1") == "1"
EMA_DECAY    = float(os.environ.get("EMA_DECAY", "0.999"))

T_0    = int(os.environ.get("T_0", "200"))
T_MULT = int(os.environ.get("T_MULT", "2"))

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
            if USE_BF16:
                inputs = inputs.to(torch.bfloat16)
            preds = model.generate(inputs, **gen_kwargs)
            text = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
            refs.extend(batch["text"])
            hyps.extend(text)
    model.train()
    return jiwer.cer(refs, hyps)


class ModelEMA:
    """Exponential moving average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].lerp_(param.data, 1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random_seed = int(os.environ.get("SEED", "42"))
    import random
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print(f"[erp1] Device: {DEVICE}")
    print(f"[erp1] epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LR}")
    print(f"[erp1] bf16={USE_BF16}, EMA={USE_EMA} (decay={EMA_DECAY})")

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, cache_dir=os.environ["HF_HOME"],
        language="zh", task="transcribe", local_files_only=True
    )
    train_ds = CantoneseASRDataset(TRAIN_JSONL, processor)
    val_ds   = CantoneseASRDataset(VAL_JSONL, processor)
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
        MODEL_NAME,
        attn_implementation="sdpa",  # SDPA for erp1 (τ-opus OFF)
        cache_dir=os.environ["HF_HOME"],
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    # Freeze encoder, train decoder only
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    apply_sparse_grad_wta(model, k_ratio=0.3, target="encoder+decoder", mode="sparse_grad")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    model.to(DEVICE)

    optimizer = AdamW(model.model.decoder.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT)
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    global_step, best_cer = 0, float("inf")
    log_lines = []

    for epoch in range(NUM_EPOCHS):
        epoch_idx = epoch + 1
        print(f"\n[erp1] Epoch {epoch_idx}/{NUM_EPOCHS}")

        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}/{NUM_EPOCHS}")

        for bidx, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BF16):
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
                if ema is not None:
                    ema.update(model)
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
        log_line = f"Epoch {epoch_idx}: avg_loss={avg_loss:.4f}, val_CER={cer*100:.2f}%, lr={scheduler.get_last_lr()[0]:.2e}, steps={global_step}"
        print(f"\n{log_line}")
        log_lines.append(log_line)

        # Save per-epoch
        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch{epoch_idx}")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))

        # Save EMA version if best
        if ema is not None and cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            ema.apply_shadow(model)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            ema.restore(model)
            print(f"  New best CER (EMA): {best_cer*100:.2f}% -> saved to {best_dir}")
        elif cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            print(f"  New best CER: {best_cer*100:.2f}% -> saved to {best_dir}")

    # Final
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))

    with open(os.path.join(OUTPUT_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n[erp1] Done. Best val CER: {best_cer*100:.2f}%")


if __name__ == "__main__":
    main()
