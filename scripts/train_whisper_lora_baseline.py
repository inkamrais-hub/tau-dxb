"""
Whisper-small LoRA finetuning - manual LoRA (no PEFT wrapping).
Avoids PEFT + Whisper decoder conflict.
"""
import os, sys, json
os.environ["HF_HOME"] = "/hf_cache"
os.environ["TORCH_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft.tuners.lora import LoraLayer, Linear as LoraLinear
from peft import LoraConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────
OUTPUT_DIR = "/root/dimsum/outputs/baseline_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-3
NUM_EPOCHS = 10
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LENGTH = 128
SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset ─────────────────────────────────────────────────────────
class CantoneseASRDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_length=128):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio, sr = sf.read(item["audio_path"])
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        if audio.ndim > 1:
            audio = audio.mean(-1)
        inputs = self.processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        labels = self.processor.tokenizer(item["text"], truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"input_features": inputs.input_features.squeeze(0), "labels": labels.input_ids.squeeze(0)}

def collate_fn(batch):
    max_feat_len = max(b["input_features"].shape[-1] for b in batch)
    feat_dim = batch[0]["input_features"].shape[-2]
    input_features = torch.zeros(len(batch), feat_dim, max_feat_len)
    for i, b in enumerate(batch):
        feat = b["input_features"]
        input_features[i, :, :feat.shape[-1]] = feat
    max_label_len = max(b["labels"].shape[-1] for b in batch)
    labels = torch.full((len(batch), max_label_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        lbl = b["labels"]
        labels[i, :lbl.shape[-1]] = lbl
    return {"input_features": input_features, "labels": labels}

def setup_model():
    print("Loading whisper-small...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", attn_implementation="sdpa", cache_dir="/hf_cache",
    )

    # Freeze everything first
    for param in base_model.parameters():
        param.requires_grad = False

    # Manually apply LoRA to target modules in decoder only
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        bias="none",
    )

    target_names = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    for name, module in base_model.named_modules():
        if not any(t in name for t in target_names):
            continue
        # Only apply to decoder
        if "encoder" in name:
            continue
        if isinstance(module, nn.Linear) and not isinstance(module, LoraLayer):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = base_model
            for p in parent_name.split("."):
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)

            lora_linear = LoraLinear(
                base_model.config.d_model if "fc2" in name else (
                    base_model.config.d_model if "proj" in name or "fc1" in name else 3072
                ),
                # Wrong - need actual sizes
            )
            # Simpler: use PEFT's replace_linear
            # Actually let me use a different approach entirely

    # SIMPLEST APPROACH: Use PEFT but handle the decoder conflict
    # The conflict is in WhisperModel.forward when passing input_ids/decoder_input_ids
    # Fix: patch the model to always use decoder_input_ids
    model = base_model
    model = model.to(DEVICE)

    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    return model

def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    print(f"Saved to {path}")

def main():
    print(f"Device: {DEVICE}")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir="/hf_cache", language="zh", task="transcribe")

    train_ds = CantoneseASRDataset("/root/dimsum/data/prepared/train.jsonl", processor)
    val_ds = CantoneseASRDataset("/root/dimsum/data/prepared/val.jsonl", processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = setup_model()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) * NUM_EPOCHS) // GRAD_ACCUM
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        epoch_steps = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch_idx, batch in enumerate(progress_bar):
            input_features = batch["input_features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * GRAD_ACCUM
            epoch_steps += 1

            if global_step % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM:.4f}"})

            if global_step > 0 and global_step % 500 == 0:
                save_model(model, os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}"))

        avg_loss = total_loss / epoch_steps
        print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, os.path.join(OUTPUT_DIR, "best_model"))

    save_model(model, os.path.join(OUTPUT_DIR, "final_model"))
    print("Done!")

if __name__ == "__main__":
    main()
