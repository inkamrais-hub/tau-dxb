"""
train_echo_opus1.py — Echo-Opus-1 训练脚本

核心设计：
  - Base: whisper-small from scratch (no warmstart)
  - τ-opus: 从 τ* 初始化，per-type σ（encoder=softplus, decoder=self=sigmoid, cross=exp）
  - Sparse: encoder=WTA 激活稀疏+可学习剪枝，decoder=sparse_grad 纯梯度稀疏
  - Triton: fused top-k WTA kernel（自动检测，不可用时 fallback PyTorch）
  - LR: 1ep warmup + CosineAnnealingWarmRestarts(T0=4ep, Tmult=2) → 15 epochs
  - 数据增强: 时间反转 10% + 噪声混合 5%（同 erp4）
  - 15 epoch, bf16, EMA, 稀疏正则化

环境变量:
  TRAIN_JSONL, VAL_JSONL, TAU_STAR_JSON, OUTPUT_DIR, MODEL_NAME
  BATCH_SIZE, GRAD_ACCUM, LR, TAU_LR, NUM_EPOCHS
  SPARSITY_LAMBDA, INIT_K_RATIO
"""
import os, sys, json, random, math

os.environ.setdefault("HF_HOME", "/hy-tmp/hf_cache")
os.environ.setdefault("TORCH_HOME", "/hy-tmp/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
    apply_sparse_grad_wta, get_sparsity_loss,
)
from triton_wta import has_triton

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
TRAIN_JSONL     = os.environ.get("TRAIN_JSONL", "/hy-tmp/dimsum/data/prepared/train.jsonl")
VAL_JSONL       = os.environ.get("VAL_JSONL", "/hy-tmp/dimsum/data/prepared/val.jsonl")
TAU_STAR_JSON   = os.environ.get("TAU_STAR_JSON", "/hy-tmp/dimsum/outputs/echo-opus-1/tau_star.json")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "/hy-tmp/dimsum/outputs/echo-opus-1")
MODEL_NAME      = os.environ.get("MODEL_NAME", "/hy-tmp/whisper-small-local")

BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "16"))
GRAD_ACCUM       = int(os.environ.get("GRAD_ACCUM", "2"))
LR               = float(os.environ.get("LR", "5e-6"))
TAU_LR           = float(os.environ.get("TAU_LR", "3e-4"))
NUM_EPOCHS       = int(os.environ.get("NUM_EPOCHS", "15"))
MAX_LENGTH       = 128
SAMPLING_RATE    = 16000
USE_BF16         = os.environ.get("USE_BF16", "1") == "1"
USE_EMA          = os.environ.get("USE_EMA", "1") == "1"
USE_FLASH        = os.environ.get("USE_FLASH", "1") == "1"  # Flash V2 Triton kernel
EMA_DECAY        = float(os.environ.get("EMA_DECAY", "0.999"))
SPARSITY_LAMBDA  = float(os.environ.get("SPARSITY_LAMBDA", "0.01"))
INIT_K_RATIO     = float(os.environ.get("INIT_K_RATIO", "0.4"))

# Cosine Warm Restarts: warmup 1ep, T0=4ep, Tmult=2
# → restarts at epoch 5, 13
T0_EPOCHS = int(os.environ.get("T0_EPOCHS", "4"))
T_MULT    = int(os.environ.get("T_MULT", "2"))

SIGMA_MAP = {
    "encoder_self": "softplus",
    "decoder_self": "sigmoid",
    "decoder_cross": "exp",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CuDNN optimization for fixed-shape inputs
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ═══════════════════════════════════════════════════════════════
# Dataset（同 erp4 数据增强）
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
            # 时间反转 10%
            if random.random() < 0.1:
                audio = audio[::-1].copy()
            # 噪声混合 5%（与同类样本混合）
            if random.random() < 0.05 and len(self.data) > 1:
                other_idx = random.randint(0, len(self.data) - 1)
                if other_idx != idx:
                    other_audio = self._load_audio(self.data[other_idx]["audio_path"])
                    min_len = min(len(audio), len(other_audio))
                    if min_len > 0:
                        audio = 0.9 * audio[:min_len] + 0.1 * other_audio[:min_len]

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
    
    # Fix the generation config outdated ValueError for older weights
    if hasattr(model, "generation_config"):
        model.generation_config.language = "zh"
        model.generation_config.task = "transcribe"
        
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
    # τ stats
    tau_stats = {}
    for name, module in model.named_modules():
        if hasattr(module, "_stau_opus") and module._stau_opus is not None:
            opus = module._stau_opus
            taus = (F.softplus(opus.log_tau) + 1.0).detach().float().cpu().tolist()
            tau_stats[name] = {"mean": sum(taus)/len(taus), "min": min(taus), "max": max(taus)}
    return cer, tau_stats


def collect_k_ratios():
    """收集所有层的 k_ratio。"""
    from patch_whisper_sparse_grad import _LEARNABLE_K
    k_info = {}
    for name, lk in _LEARNABLE_K.items():
        k_info[name] = {
            "k_ratio": round(lk.k_ratio, 4),
            "logit_k": round(lk.logit_k.item(), 4),
        }
    return k_info


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
# LR Schedule: warmup(1ep) + CosineAnnealingWarmRestarts(T0, Tmult)
# ═══════════════════════════════════════════════════════════════

def make_lr_lambda(warmup_steps: int, t0_steps: int, t_mult: int):
    def fn(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        # Cosine annealing with warm restarts
        cycle_step = step - warmup_steps
        cycle_len = t0_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            cycle_len *= t_mult
        if cycle_len <= 0:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * cycle_step / cycle_len))
    return fn


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    print(f"[echo-opus-1] ========================================")
    print(f"[echo-opus-1] Device: {DEVICE}")
    print(f"[echo-opus-1] Triton available: {has_triton()}")
    print(f"[echo-opus-1] epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")
    print(f"[echo-opus-1] lr={LR}, tau_lr={TAU_LR}, bf16={USE_BF16}, EMA={USE_EMA}, flash={USE_FLASH}")
    print(f"[echo-opus-1] σ: {SIGMA_MAP}")
    print(f"[echo-opus-1] learnable k_ratio: init={INIT_K_RATIO}, sparsity_λ={SPARSITY_LAMBDA}")
    print(f"[echo-opus-1] LR schedule: warmup(1ep) + CosineWarmRestarts(T0={T0_EPOCHS}ep, Tmult={T_MULT})")
    print(f"[echo-opus-1] τ init from: {TAU_STAR_JSON}")
    print(f"[echo-opus-1] output: {OUTPUT_DIR}")
    print(f"[echo-opus-1] ========================================")

    # 1. Processor
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, cache_dir=os.environ["HF_HOME"],
        language="zh", task="transcribe", local_files_only=True
    )

    # 2. Model (eager for τ-opus)
    dtype = torch.bfloat16 if USE_BF16 else torch.float32
    print(f"[echo-opus-1] Loading whisper-small (eager, {dtype}) ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        cache_dir=os.environ["HF_HOME"],
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )

    # 3. 全参解冻
    for p in model.parameters():
        p.requires_grad = True

    # 4. τ-opus（从 τ* 初始化）
    print(f"[echo-opus-1] Applying learnable τ-opus (init from τ*, flash={USE_FLASH}) ...")
    apply_learnable_stau(
        model, tau_init=TAU_STAR_JSON, alpha_init=1.0, sigma_name=SIGMA_MAP,
        use_flash=USE_FLASH,
    )

    # 5. Sparse — encoder=WTA, decoder=sparse_grad only（不剪枝）
    print(f"[echo-opus-1] Applying sparse patches ...")
    print(f"  encoder: wta_activation (competitive sparse + learnable pruning)")
    print(f"  decoder: sparse_grad (gradient sparsity only, NO activation pruning)")
    apply_sparse_grad_wta(
        model,
        target="encoder+decoder",
        encoder_mode="wta_activation",
        decoder_mode="sparse_grad",
        learnable_k=True,
        init_k_ratio=INIT_K_RATIO,
        track_wins=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[echo-opus-1] Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    model.to(DEVICE)

    # 5b. torch.compile encoder only (safer with hooks)
    if os.environ.get("USE_COMPILE", "0") == "1":
        try:
            model.model.encoder = torch.compile(model.model.encoder, mode="reduce-overhead")
            print(f"[echo-opus-1] torch.compile ON (encoder only, mode=reduce-overhead)")
        except Exception as e:
            print(f"[echo-opus-1] torch.compile skipped: {e}")
    else:
        print(f"[echo-opus-1] torch.compile OFF")

    # 初始 k_ratio 快照
    init_k_info = collect_k_ratios()
    init_k_mean = sum(v["k_ratio"] for v in init_k_info.values()) / max(1, len(init_k_info))
    print(f"[echo-opus-1] Initial k_ratio mean: {init_k_mean:.4f}")
    with open(os.path.join(OUTPUT_DIR, "k_ratio_init.json"), "w", encoding="utf-8") as f:
        json.dump(init_k_info, f, ensure_ascii=False, indent=2)

    # 6. Data
    train_ds = CantoneseASRDataset(TRAIN_JSONL, processor, augment=True)
    val_ds   = CantoneseASRDataset(VAL_JSONL, processor, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"[echo-opus-1] Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 7. Optimizer (base + τ/α 分组)
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

    # LR Schedule: warmup(1ep) + CosineWarmRestarts(T0=4, Tmult=2)
    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    warmup_steps = steps_per_epoch
    t0_steps = T0_EPOCHS * steps_per_epoch
    scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(warmup_steps, t0_steps, T_MULT))
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    # 8. Train
    global_step, best_cer = 0, float("inf")
    log_lines = []
    k_ratio_history = []
    start_epoch = 1

    # Resume from checkpoint
    resume_path = os.environ.get("RESUME_CKPT", "")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_cer = ckpt["best_cer"]
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
        if "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        if "log_lines" in ckpt:
            log_lines = ckpt["log_lines"]
        if "k_ratio_history" in ckpt:
            k_ratio_history = ckpt["k_ratio_history"]
        if ema is not None and ckpt.get("ema_shadow"):
            ema.shadow = ckpt["ema_shadow"]
            print(f"[echo-opus-1]   EMA shadow restored ({len(ema.shadow)} params)")
        print(f"[echo-opus-1] Resumed from {resume_path}: epoch={start_epoch}, step={global_step}, best_cer={best_cer*100:.2f}%")

    for epoch in range(start_epoch - 1, NUM_EPOCHS):
        ep = epoch + 1
        print(f"\n[echo-opus-1] Epoch {ep}/{NUM_EPOCHS}")

        model.train()
        total_loss = 0.0
        total_ctc = 0.0
        total_sparse_reg = 0.0
        n_steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{NUM_EPOCHS}")

        for bidx, batch in enumerate(pbar):
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                    ctc = model(
                        input_features=batch["input_features"].to(DEVICE),
                        labels=batch["labels"].to(DEVICE),
                    ).loss / GRAD_ACCUM

                    sparsity_reg = get_sparsity_loss().to(ctc.device)
                    loss = ctc + SPARSITY_LAMBDA * sparsity_reg

                loss.backward()
            except RuntimeError as e:
                print(f"\n[WARN] batch {bidx} failed: {e}, skipping")
                optimizer.zero_grad()
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARN] batch {bidx} NaN/Inf loss, skipping")
                optimizer.zero_grad()
                continue

            if (bidx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
                global_step += 1

            total_loss += loss.item() * GRAD_ACCUM
            total_ctc += ctc.item() * GRAD_ACCUM
            total_sparse_reg += sparsity_reg.item() * GRAD_ACCUM
            n_steps += 1
            if global_step % 10 == 0 and global_step > 0:
                pbar.set_postfix({
                    "ctc": f"{ctc.item()*GRAD_ACCUM:.4f}",
                    "sp": f"{sparsity_reg.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })

        # flush 剩余梯度（末 batch 不完整时）
        if n_steps > 0 and (bidx + 1) % GRAD_ACCUM != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            global_step += 1

        avg_loss = total_loss / n_steps
        avg_ctc = total_ctc / n_steps
        avg_sparse = total_sparse_reg / n_steps

        # k_ratio evolution
        k_info = collect_k_ratios()
        k_mean = sum(v["k_ratio"] for v in k_info.values()) / max(1, len(k_info))
        k_ratio_history.append({"epoch": ep, "mean_k_ratio": round(k_mean, 4), "layers": k_info})
        with open(os.path.join(OUTPUT_DIR, f"k_ratio_epoch{ep}.json"), "w", encoding="utf-8") as f:
            json.dump(k_info, f, ensure_ascii=False, indent=2)

        # CER eval
        cer, tau_stats = evaluate_cer(model, processor, val_loader)
        tau_means = [s["mean"] for s in tau_stats.values()]
        tau_avg = sum(tau_means)/len(tau_means) if tau_means else 0

        log_line = (f"Epoch {ep:2d}: ctc={avg_ctc:.4f}, sparse={avg_sparse:.4f}, "
                    f"loss={avg_loss:.4f}, CER={cer*100:.2f}%, "
                    f"k={k_mean:.4f}, tau_avg={tau_avg:.2f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}, step={global_step}")
        print(f"\n{log_line}")
        log_lines.append(log_line)

        # τ checkpoint
        if tau_stats:
            with open(os.path.join(OUTPUT_DIR, f"tau_epoch{ep}.json"), "w", encoding="utf-8") as f:
                json.dump(tau_stats, f, ensure_ascii=False, indent=2)

        # Save (resume-capable)
        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch{ep}")
        os.makedirs(epoch_dir, exist_ok=True)
        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "best_cer": best_cer,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "log_lines": log_lines,
            "k_ratio_history": k_ratio_history,
            "ema_shadow": (ema.shadow if ema is not None else None),
        }
        torch.save(ckpt, os.path.join(epoch_dir, "ckpt.pt"))
        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))

        # Best (EMA)
        if ema is not None and cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            ema.apply_shadow(model)
            try:
                torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            finally:
                ema.restore(model)
            print(f"  New best CER (EMA): {best_cer*100:.2f}%")
        elif cer < best_cer:
            best_cer = cer
            best_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
            print(f"  New best CER: {best_cer*100:.2f}%")

        # ─── 每 epoch 自动上传产物到 atomgit ───
        # 小文件 (tau/k_ratio/log/metadata) push 到 git，大文件 (model.pt) 只记录 metadata
        # 失败不阻断训练
        try:
            import subprocess as _sp
            _push_script = os.path.join(SCRIPT_DIR, "remote_git_push.py")
            if os.path.exists(_push_script):
                _r = _sp.run([sys.executable, _push_script, str(ep),
                              "--output-dir", OUTPUT_DIR],
                             capture_output=True, timeout=180, text=True)
                if _r.returncode == 0:
                    print(f"  [git-push] epoch {ep} uploaded ✓")
                else:
                    print(f"  [git-push] epoch {ep} warning: {_r.stderr[:200]}")
        except Exception as _e:
            print(f"  [git-push] epoch {ep} failed (训练继续): {_e}")

    # 9. Final save
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))

    with open(os.path.join(OUTPUT_DIR, "k_ratio_history.json"), "w", encoding="utf-8") as f:
        json.dump(k_ratio_history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n[echo-opus-1] ========================================")
    print(f"[echo-opus-1] Done. Best CER: {best_cer*100:.2f}%")
    print(f"[echo-opus-1] k_ratio history: {OUTPUT_DIR}/k_ratio_history.json")
    print(f"[echo-opus-1] ========================================")

    # Final τ
    final_taus = {}
    for name, module in model.named_modules():
        if hasattr(module, "_stau_opus") and module._stau_opus is not None:
            opus = module._stau_opus
            sigma_type = getattr(opus, "sigma_name", "unknown")
            taus = (F.softplus(opus.log_tau) + 1.0).detach().float().cpu().tolist()
            final_taus[name] = {
                "sigma": sigma_type,
                "mean": round(sum(taus)/len(taus), 4),
                "min": round(min(taus), 4),
                "max": round(max(taus), 4),
                "raw": [round(t, 4) for t in taus],
            }
    with open(os.path.join(OUTPUT_DIR, "final_tau.json"), "w", encoding="utf-8") as f:
        json.dump(final_taus, f, ensure_ascii=False, indent=2)
    print(f"[echo-opus-1] Final τ: {OUTPUT_DIR}/final_tau.json")


if __name__ == "__main__":
    main()
