"""
Train erp4 — τ-opus 精细微调，从 erp3 epoch5 warm start.

与 erp3 的关键区别：
  - Warm start 从 erp3 epoch5（不从 erp1）
  - 不继续剪枝（冻结 erp3 已剪枝状态）
  - 更低的 LR：base=5e-6, τ/α=3e-4
  - 固定 k_ratio（encoder=0.4, decoder=0.3）
  - 7 epoch，简单余弦退火
  - τ 保持可学习（不回火到 1）
"""
import os, sys, json, random, math

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/hy-tmp/hf_cache")
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/hy-tmp/hf_cache")
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
import librosa
import jiwer
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from patch_whisper_stau import apply_learnable_stau
from patch_whisper_sparse_grad import (
    apply_sparse_grad_wta, set_sparse_grad_k_ratio,
)

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", "/hy-tmp/dimsum/data/prepared/train.jsonl")
VAL_JSONL   = os.environ.get("VAL_JSONL", "/hy-tmp/dimsum/data/prepared/val.jsonl")
ERP3_CKPT   = os.environ.get("ERP3_CKPT", "/hy-tmp/dimsum/outputs/erp3/epoch5/model.pt")
TAU_STAR_JSON = os.environ.get("TAU_STAR_JSON", "/hy-tmp/dimsum/outputs/erp2_tau_star.json")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "/hy-tmp/dimsum/outputs/erp4")
MODEL_NAME  = os.environ.get("MODEL_NAME", "/hy-tmp/whisper-small-local")

BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))
LR           = float(os.environ.get("LR", "5e-6"))      # 全参，比 erp3 低 6x
TAU_LR       = float(os.environ.get("TAU_LR", "3e-4"))  # τ/α 维持可学
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "7"))
MAX_LENGTH   = 128
SAMPLING_RATE = 16000
USE_BF16     = os.environ.get("USE_BF16", "1") == "1"
USE_EMA      = os.environ.get("USE_EMA", "1") == "1"
EMA_DECAY    = float(os.environ.get("EMA_DECAY", "0.999"))

# 固定 k_ratio（不再课程式变化）
ENC_K_RATIO = float(os.environ.get("ENC_K_RATIO", "0.4"))
DEC_K_RATIO = float(os.environ.get("DEC_K_RATIO", "0.3"))

# 数据增强（轻量，保持泛化）
P_TIME_REVERSE = float(os.environ.get("P_TIME_REVERSE", "0.1"))
P_MIXSPEECH    = float(os.environ.get("P_MIXSPEECH", "0.05"))
MIXSPEECH_ALPHA = float(os.environ.get("MIXSPEECH_ALPHA", "0.9"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 分类型 σ 配置（和 erp2/erp3 一致）
SIGMA_MAP = {
    "encoder_self": "softplus",
    "decoder_self": "sigmoid",
    "decoder_cross": "exp",
}


# ═══════════════════════════════════════════════════════════════
# Dataset（同 erp3）
# ═══════════════════════════════════════════════════════════════

class CantoneseASRDataset(Dataset):
    def __init__(self, jsonl_path, processor, augment=False):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.processor = processor
        self.augment = augment

    def __len__(self): return len(self.data)

    def _load_audio(self, path):
        audio, sr = sf.read(path)
        if sr != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        if audio.ndim > 1:
            audio = audio.mean(-1)
        return audio.astype("float32")

    def __getitem__(self, idx):
        item = self.data[idx]
        audio = self._load_audio(item["audio_path"])
        text = item["text"]

        if self.augment:
            if random.random() < P_TIME_REVERSE:
                audio = audio[::-1].copy()
            if random.random() < P_MIXSPEECH and len(self.data) > 1:
                other_idx = random.randint(0, len(self.data) - 1)
                if other_idx != idx:
                    other_audio = self._load_audio(self.data[other_idx]["audio_path"])
                    min_len = min(len(audio), len(other_audio))
                    if min_len > 0:
                        audio = MIXSPEECH_ALPHA * audio[:min_len] + (1 - MIXSPEECH_ALPHA) * other_audio[:min_len]

        inp = self.processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        lbl = self.processor.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {
            "input_features": inp.input_features.squeeze(0),
            "labels": lbl.input_ids.squeeze(0),
            "text": text,
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
    return {"input_features": inp, "labels": lbl, "text": [b["text"] for b in batch]}


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_cer(model, processor, val_loader):
    model.eval()
    refs, hyps = [], []
    gen_kwargs = {"language": "zh", "task": "transcribe", "num_beams": 1, "do_sample": False}
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
    cer = jiwer.cer(refs, hyps)
    # τ 统计
    tau_stats = {}
    for name, module in model.named_modules():
        if hasattr(module, "_stau_opus") and module._stau_opus is not None:
            opus = module._stau_opus
            taus = (F.softplus(opus.log_tau) + 1.0).detach().float().cpu().tolist()
            tau_stats[name] = {"mean": sum(taus)/len(taus), "min": min(taus), "max": max(taus)}
    return cer, tau_stats


# ═══════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════

class ModelEMA:
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


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    print(f"[erp4] Device: {DEVICE}")
    print(f"[erp4] epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"[erp4] lr={LR}, tau_lr={TAU_LR}, bf16={USE_BF16}, EMA={USE_EMA}")
    print(f"[erp4] k_ratio: encoder={ENC_K_RATIO}, decoder={DEC_K_RATIO} (fixed)")
    print(f"[erp4] no pruning (frozen from erp3)")
    print(f"[erp4] σ: {SIGMA_MAP}")

    # 1. Processor
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, cache_dir=os.environ["HF_HOME"],
        language="zh", task="transcribe", local_files_only=True
    )

    # 2. Model (eager for τ-opus)
    print(f"[erp4] Loading model (bf16, eager) ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        cache_dir=os.environ["HF_HOME"],
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )

    # 3. 全参解冻
    for p in model.parameters():
        p.requires_grad = True

    # 4. τ-opus（和 erp3 同样的初始化方式，权重稍后被 erp3 checkpoint 覆盖）
    print(f"[erp4] Applying τ-opus ...")
    apply_learnable_stau(
        model,
        tau_init=TAU_STAR_JSON,
        alpha_init=1.0,
        sigma_name=SIGMA_MAP,
    )

    # 5. SparseGrad + WTA（不追踪 win count，不剪枝）
    print(f"[erp4] Applying sparse patches ...")
    apply_sparse_grad_wta(
        model,
        target="encoder+decoder",
        encoder_mode="wta_activation",
        decoder_mode="sparse_grad",
        encoder_k_ratio=ENC_K_RATIO,
        decoder_k_ratio=DEC_K_RATIO,
        track_wins=False,  # 不统计 win —— 不剪枝
    )

    # 6. 加载 erp3 epoch5 权重（warm start）
    print(f"[erp4] Loading erp3 weights from {ERP3_CKPT} ...")
    state_dict = torch.load(ERP3_CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    model.to(DEVICE)

    # 7. Data
    train_ds = CantoneseASRDataset(TRAIN_JSONL, processor, augment=True)
    val_ds   = CantoneseASRDataset(VAL_JSONL, processor, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 8. Optimizer (base + τ/α 分组)
    base_params, tau_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "log_tau" in name or "log_alpha" in name:
            tau_params.append(param)
        else:
            base_params.append(param)
    optimizer = AdamW([
        {"params": base_params, "lr": LR},
        {"params": tau_params, "lr": TAU_LR},
    ], weight_decay=0.01)

    # 简单余弦退火：1 epoch 线性预热 → 6 epoch 余弦衰减
    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    warmup_steps = steps_per_epoch  # 第一个 epoch 线性预热
    total_steps = steps_per_epoch * NUM_EPOCHS

    def _lr_lambda(step):
        if step < warmup_steps:
            return 0.5 * step / warmup_steps  # 0 → 0.5
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    # 9. Train
    global_step, best_cer = 0, float("inf")
    log_lines = []

    for epoch in range(NUM_EPOCHS):
        ep = epoch + 1
        print(f"\n[erp4] Epoch {ep}/{NUM_EPOCHS}")

        # 固定 k_ratio（不课程衰减，不剪枝）
        set_sparse_grad_k_ratio(model, ENC_K_RATIO, layer_type="encoder")
        set_sparse_grad_k_ratio(model, DEC_K_RATIO, layer_type="decoder")

        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{NUM_EPOCHS}")

        for bidx, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BF16):
                loss = model(
                    input_features=batch["input_features"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                ).loss / GRAD_ACCUM

            loss.backward()

            if (bidx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
                global_step += 1

            total_loss += loss.item() * GRAD_ACCUM
            n_steps += 1
            if global_step % 10 == 0 and global_step > 0:
                pbar.set_postfix({
                    "loss": f"{loss.item()*GRAD_ACCUM:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })

        avg_loss = total_loss / n_steps
        cer, tau_stats = evaluate_cer(model, processor, val_loader)
        tau_means = [s["mean"] for s in tau_stats.values()]
        tau_avg = sum(tau_means)/len(tau_means) if tau_means else 0

        log_line = (f"Epoch {ep}: avg_loss={avg_loss:.4f}, val_CER={cer*100:.2f}%, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}, "
                    f"tau_avg={tau_avg:.2f}, steps={global_step}")
        print(f"\n{log_line}")
        log_lines.append(log_line)

        # 保存 τ 分布
        if tau_stats:
            with open(os.path.join(OUTPUT_DIR, f"tau_epoch{ep}.json"), "w", encoding="utf-8") as f:
                json.dump(tau_stats, f, ensure_ascii=False, indent=2)

        # Save
        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch{ep}")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))

        if ema is not None and cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            ema.apply_shadow(model)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            ema.restore(model)
            print(f"  New best CER (EMA): {best_cer*100:.2f}%")
        elif cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            print(f"  New best CER: {best_cer*100:.2f}%")

    # Final
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))

    with open(os.path.join(OUTPUT_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n[erp4] Done. Best val CER: {best_cer*100:.2f}%")

    # 记录最终 τ 值
    final_taus = {}
    for name, module in model.named_modules():
        if hasattr(module, "_stau_opus") and module._stau_opus is not None:
            opus = module._stau_opus
            sigma_type = getattr(opus, "sigma_type", "unknown")
            taus = (F.softplus(opus.log_tau) + 1.0).detach().float().cpu().tolist()
            final_taus[name] = {
                "sigma": sigma_type,
                "mean": sum(taus)/len(taus),
                "min": min(taus), "max": max(taus),
                "raw": taus,
            }
    with open(os.path.join(OUTPUT_DIR, "final_tau.json"), "w", encoding="utf-8") as f:
        json.dump(final_taus, f, ensure_ascii=False, indent=2)
    print(f"Final τ logged to {OUTPUT_DIR}/final_tau.json")


if __name__ == "__main__":
    main()
