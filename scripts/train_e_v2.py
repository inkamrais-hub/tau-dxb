"""
Train E-v2: whisper-small Cantonese ASR with
  - learnable τ-opus (initialized near softmax, τ≈1.0)
  - SparseGrad-WTA curriculum (k_ratio 0.5 → 0.4 → 0.3)
  - time-reversal augmentation (50%)
  - MixSpeech-style acoustic mixup (10%)
  - CosineAnnealingWarmRestarts cyclic LR schedule

Designed for remote GPU server paths; falls back to local smoke-test paths.
"""
import os
import sys
import json
import random
import math

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/hf_cache")
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/hf_cache")
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

# Make patch modules importable from the same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from patch_whisper_stau import apply_learnable_stau
from patch_whisper_sparse_grad import apply_sparse_grad_wta, set_sparse_grad_k_ratio

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", "/root/dimsum/data/prepared/train.jsonl")
VAL_JSONL   = os.environ.get("VAL_JSONL", "/root/dimsum/data/prepared/val.jsonl")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "/root/dimsum/outputs/e_v2")

BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))
LR           = float(os.environ.get("LR", "5e-5"))
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "10"))
MAX_LENGTH   = 128
SAMPLING_RATE = 16000

# Augmentation
P_TIME_REVERSAL = 0.5
P_MIXSPEECH     = 0.1
MIX_ALPHA       = 0.1

# SparseGrad curriculum
SPARSE_K_SCHEDULE = {
    1: 0.50,
    2: 0.50,
    3: 0.40,
    4: 0.40,
}

# LR schedule: fast cycles early, slower later
T_0    = int(os.environ.get("T_0", "200"))
T_MULT = int(os.environ.get("T_MULT", "2"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# Dataset: time reversal + acoustic mixup
# ═══════════════════════════════════════════════════════════════

class CantoneseASRDataset(Dataset):
    def __init__(self, jsonl_path, processor, augment=True):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def _load_audio(self, path):
        audio, sr = sf.read(path)
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        if audio.ndim > 1:
            audio = audio.mean(-1)
        return audio.astype("float32")

    def _mix_pair(self, audio_a, audio_b, alpha):
        """Mix two waveforms to the same length by padding/truncating."""
        target_len = max(len(audio_a), len(audio_b))
        def pad_or_cut(x):
            if len(x) < target_len:
                return torch.cat([x, torch.zeros(target_len - len(x), dtype=x.dtype)])
            return x[:target_len]
        a = pad_or_cut(torch.from_numpy(audio_a))
        b = pad_or_cut(torch.from_numpy(audio_b))
        mixed = (1 - alpha) * a + alpha * b
        return mixed.numpy()

    def __getitem__(self, idx):
        item = self.data[idx]
        audio = self._load_audio(item["audio_path"])
        text = item["text"]

        # Time reversal: audio + text both reversed
        if self.augment and random.random() < P_TIME_REVERSAL:
            audio = audio[::-1].copy()
            text = text[::-1]

        # Acoustic MixSpeech: mix with a random partner, keep primary label
        if self.augment and random.random() < P_MIXSPEECH:
            partner = self.data[random.randint(0, len(self.data) - 1)]
            audio_b = self._load_audio(partner["audio_path"])
            audio = self._mix_pair(audio, audio_b, MIX_ALPHA)

        inp = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        )
        lbl = self.processor.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        return {
            "input_features": inp.input_features.squeeze(0),
            "labels": lbl.input_ids.squeeze(0),
            "text": text,
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


# ═══════════════════════════════════════════════════════════════
# Model setup
# ═══════════════════════════════════════════════════════════════

def setup_model():
    print("Loading whisper-small ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        attn_implementation="eager",  # required by τ-opus
        cache_dir=os.environ["HF_HOME"],
    )

    # Freeze encoder; decoder (including learnable τ) will be trained
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    print("Applying learnable τ-opus (init τ≈1.0) ...")
    apply_learnable_stau(model, tau_init=1.0, alpha_init=1.0)

    print("Applying SparseGrad-WTA ...")
    apply_sparse_grad_wta(model, k_ratio=0.5, target="encoder+decoder", mode="sparse_grad")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# SparseGrad curriculum
# ═══════════════════════════════════════════════════════════════

def get_k_ratio_for_epoch(epoch_1based: int) -> float:
    if epoch_1based in SPARSE_K_SCHEDULE:
        return SPARSE_K_SCHEDULE[epoch_1based]
    return 0.30


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    print(f"[E-v2] Device: {DEVICE}")
    print(f"[E-v2] epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LR}")
    print(f"[E-v2] T_0={T_0}, T_mult={T_MULT}")
    print(f"[E-v2] augmentation: time_reversal={P_TIME_REVERSAL}, mixspeech={P_MIXSPEECH} (α={MIX_ALPHA})")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        cache_dir=os.environ["HF_HOME"],
        language="zh",
        task="transcribe",
    )

    train_ds = CantoneseASRDataset(TRAIN_JSONL, processor, augment=True)
    val_ds   = CantoneseASRDataset(VAL_JSONL,   processor, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")

    model = setup_model()

    # Optimizer covers decoder params + learnable τ/α params
    optimizer = AdamW(model.model.decoder.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT)

    global_step, best_cer = 0, float("inf")
    log_lines = []

    for epoch in range(NUM_EPOCHS):
        epoch_idx = epoch + 1
        k_ratio = get_k_ratio_for_epoch(epoch_idx)
        set_sparse_grad_k_ratio(model, k_ratio)
        print(f"\n[E-v2] Epoch {epoch_idx}/{NUM_EPOCHS} — SparseGrad k_ratio={k_ratio:.2f}")

        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}/{NUM_EPOCHS}")

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

        # Report average tau values across decoder attention heads
        tau_vals, alpha_vals = [], []
        for module in model.model.decoder.modules():
            if hasattr(module, "_stau_opus") and module._stau_opus is not None:
                with torch.no_grad():
                    tau = (torch.nn.functional.softplus(module._stau_opus.log_tau) + 1.0).cpu()
                    alpha = torch.exp(module._stau_opus.log_alpha).cpu()
                tau_vals.append(tau.mean().item())
                alpha_vals.append(alpha.mean().item())
        tau_mean = sum(tau_vals) / len(tau_vals) if tau_vals else 0.0
        alpha_mean = sum(alpha_vals) / len(alpha_vals) if alpha_vals else 0.0

        log_line = (
            f"Epoch {epoch_idx}: avg_loss={avg_loss:.4f}, val_CER={cer*100:.2f}%, "
            f"lr={scheduler.get_last_lr()[0]:.2e}, steps={global_step}, "
            f"tau_mean={tau_mean:.3f}, alpha_mean={alpha_mean:.3f}"
        )
        print(f"\n{log_line}")
        log_lines.append(log_line)

        # Save per-epoch checkpoint
        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch{epoch_idx}")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))

        if cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            print(f"  New best CER: {best_cer*100:.2f}% -> saved to {best_dir}")

    # Final model
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))

    with open(os.path.join(OUTPUT_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n[E-v2] Done. Best val CER: {best_cer*100:.2f}%")


if __name__ == "__main__":
    main()
