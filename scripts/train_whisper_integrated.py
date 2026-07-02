"""
Whisper-small Cantonese ASR - integrated: SparseGrad-WTA + τ-opus.
Clean patches: SparseGrad wraps activation_fn, τ-opus replaces eager_attention_forward.
"""
import os, sys, json
os.environ["HF_HOME"] = "/hf_cache"
os.environ["TORCH_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module
import soundfile as sf
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────
TAU_STAR_PATH = "/root/dimsum/outputs/whisper_small_tau_star.json"
OUTPUT_DIR = "/root/dimsum/outputs/integrated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 1e-5  # reduced for τ-opus + SparseGrad stability
NUM_EPOCHS = 5
MAX_LENGTH = 128
SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPARSE_K_RATIO = 0.3

# ═══════════════════════════════════════════════════════════════
# 1. SparseGrad-WTA
# ═══════════════════════════════════════════════════════════════

class SparseGradWTAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        _, topk_idx = x.topk(k, dim=-1)
        ctx.save_for_backward(topk_idx)
        ctx.x_shape = x.shape
        return x  # forward = identity

    @staticmethod
    def backward(ctx, grad_output):
        topk_idx, = ctx.saved_tensors
        mask = torch.zeros(ctx.x_shape, device=grad_output.device, dtype=grad_output.dtype)
        mask.scatter_(-1, topk_idx, 1.0)
        return grad_output * mask, None


class SparseGradWTAModule(nn.Module):
    """Wraps an activation function with SparseGrad-WTA."""
    def __init__(self, activation_fn, k_ratio=0.3):
        super().__init__()
        self.act_fn = activation_fn
        self.k_ratio = k_ratio

    def forward(self, x):
        x = self.act_fn(x)
        if self.training and self.k_ratio < 1.0:
            d_ff = x.shape[-1]
            k = max(1, int(d_ff * self.k_ratio))
            x = SparseGradWTAFn.apply(x, k)
        return x


def apply_sparse_grad_wta(model, k_ratio=0.3):
    """Replace activation_fn in encoder/decoder layers with SparseGrad-WTA wrapper."""
    from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer, WhisperDecoderLayer

    count = 0
    for module in model.modules():
        if isinstance(module, (WhisperEncoderLayer, WhisperDecoderLayer)):
            old_act = module.activation_fn
            module.activation_fn = SparseGradWTAModule(old_act, k_ratio)
            count += 1
    print(f"  SparseGrad-WTA applied to {count} layers (k={k_ratio})")


# ═══════════════════════════════════════════════════════════════
# 2. τ-opus — correct implementation (inlined from stau_opus.py)
# ═══════════════════════════════════════════════════════════════

CLAMP_MIN = 1e-8

# σ functions
def _sigma_softplus(x): return F.softplus(x).clamp(min=CLAMP_MIN)
def _sigma_sigmoid(x):  return torch.sigmoid(x).clamp(min=CLAMP_MIN)
def _sigma_exp(x):      return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)
def _sigma_relu(x):     return F.relu(x).clamp(min=CLAMP_MIN)
def _sigma_tanh_shift(x): return (torch.tanh(x) + 1).clamp(min=CLAMP_MIN)

_SIGMA = {
    "softplus": _sigma_softplus, "sigmoid": _sigma_sigmoid,
    "exp": _sigma_exp, "relu": _sigma_relu, "tanh_shift": _sigma_tanh_shift,
}

def _sigma_prime_softplus(x): return torch.sigmoid(x)
def _sigma_prime_sigmoid(x):  s = torch.sigmoid(x); return s * (1 - s)
def _sigma_prime_exp(x):      return torch.exp(x.clamp(max=15))
def _sigma_prime_tanh(x):     return 1 - torch.tanh(x) ** 2
def _sigma_prime_relu(x):     return (x > 0).float()

_SIGMA_PRIME = {
    "softplus": _sigma_prime_softplus, "sigmoid": _sigma_prime_sigmoid,
    "exp": _sigma_prime_exp, "tanh_shift": _sigma_prime_tanh, "relu": _sigma_prime_relu,
}

class STauOpusMaxStableFn(torch.autograd.Function):
    """τ-opus: σ(s)^τ / Σσ(s)^τ — closed-form backward with max-cancellation."""
    @staticmethod
    def forward(ctx, scores, tau, sigma_name):
        sigma_fn = _SIGMA[sigma_name]
        x_stable = scores - scores.max(dim=-1, keepdim=True).values
        sigma_val = sigma_fn(x_stable)
        q = sigma_val.clamp(min=CLAMP_MIN).pow(tau)
        sum_q = q.sum(dim=-1, keepdim=True).clamp(min=CLAMP_MIN)
        attn = q / sum_q
        ctx.save_for_backward(scores, x_stable, sigma_val, attn, sum_q)
        ctx.tau = tau
        ctx.sigma_name = sigma_name
        return attn

    @staticmethod
    def backward(ctx, dO):
        scores, x_stable, sigma_val, attn, sum_q = ctx.saved_tensors
        tau, sigma_name = ctx.tau, ctx.sigma_name
        sigma_prime_fn = _SIGMA_PRIME.get(sigma_name, lambda x: torch.ones_like(x))
        sigma_tau_m1 = sigma_val.clamp(min=CLAMP_MIN).pow(tau - 1.0)
        sigma_p = sigma_prime_fn(x_stable)
        A = tau * sigma_tau_m1 * sigma_p
        S = sum_q
        q = sigma_val.clamp(min=CLAMP_MIN).pow(tau)
        wda = (dO * q).sum(dim=-1, keepdim=True)
        da = (dO * A).sum(dim=-1, keepdim=True)
        sA = A.sum(dim=-1, keepdim=True)
        term1 = A * (dO / S - wda / S.pow(2))
        term2_full = da / S - wda * sA / S.pow(2)
        argmax = scores.argmax(dim=-1, keepdim=True)
        term2 = torch.zeros_like(term1)
        term2.scatter_(-1, argmax, -term2_full)
        dScores = term1 + term2
        return dScores.float(), None, None

def load_tau_star(path):
    with open(path, "r") as f:
        data = json.load(f)
    result = {"encoder_self": [], "decoder_self": [], "decoder_cross": []}
    for item in data:
        key = "decoder_cross" if item["cross_attention"] else (
            "decoder_self" if item["is_decoder"] else "encoder_self"
        )
        result[key].append(item["tau_per_head"])
    for k in result:
        print(f"  {k}: {len(result[k])} layers")
    return result


def _make_stau_forward():
    def stau_eager_forward(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
        if scaling is None:
            scaling = query.size(-1) ** -0.5
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        tau = getattr(module, "_stau_tau", None)
        if tau is None:
            tau = torch.tensor(1.0, device=scores.device)
        tau = tau.to(scores.device)
        sigma = getattr(module, "_stau_sigma", "softplus")

        attn_weights = STauOpusMaxStableFn.apply(scores.float(), tau, sigma)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights.to(value.dtype), value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights.to(value.dtype)
    return stau_eager_forward


def apply_tau_opus(model, tau_star_dict, sigma_map):
    enc_taus = tau_star_dict.get("encoder_self", [[1.0]*12]*12)
    dec_self_taus = tau_star_dict.get("decoder_self", [[1.0]*12]*12)
    dec_cross_taus = tau_star_dict.get("decoder_cross", [[1.0]*12]*12)

    idx_map = {"encoder_self": 0, "decoder_self": 0, "decoder_cross": 0}
    tau_map = {"encoder_self": enc_taus, "decoder_self": dec_self_taus, "decoder_cross": dec_cross_taus}

    for name, module in model.named_modules():
        if module.__class__.__name__ != "WhisperAttention":
            continue
        kind = ("decoder_cross" if ("encoder_attn" in name or "cross" in name.lower()) else
                ("decoder_self" if getattr(module, "is_decoder", False) else "encoder_self"))

        tau_list = tau_map[kind][idx_map[kind] % len(tau_map[kind])]
        idx_map[kind] += 1

        h = module.num_heads
        tau_t = torch.tensor(tau_list[:h], dtype=torch.float32).view(1, h, 1, 1)
        module.register_buffer("_stau_tau", tau_t)
        module._stau_sigma = sigma_map.get(kind, "softplus")

    whisper_module.eager_attention_forward = _make_stau_forward()
    total = sum(idx_map.values())
    print(f"  τ-opus applied to {total} attention modules "
          f"(enc={idx_map['encoder_self']}, dec_self={idx_map['decoder_self']}, "
          f"cross={idx_map['decoder_cross']})")


# ═══════════════════════════════════════════════════════════════
# 3. Dataset
# ═══════════════════════════════════════════════════════════════

class CantoneseASRDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.processor = processor

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
        labels = self.processor.tokenizer(item["text"], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {"input_features": inputs.input_features.squeeze(0), "labels": labels.input_ids.squeeze(0)}


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
    return {"input_features": inp, "labels": lbl}


# ═══════════════════════════════════════════════════════════════
# 4. Model setup
# ═══════════════════════════════════════════════════════════════

def setup_model():
    print("Loading whisper-small...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        attn_implementation="eager",  # τ-opus needs eager mode
        cache_dir="/hf_cache",
    )

    # Freeze encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    all_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {all_train:,}/{all_total:,} ({all_train/all_total*100:.1f}%)")

    # 1. SparseGrad-WTA: wrap activation_fn
    print("Applying SparseGrad-WTA (k={})...".format(SPARSE_K_RATIO))
    apply_sparse_grad_wta(model, SPARSE_K_RATIO)

    # 2. τ-opus: load τ* and replace attention
    print("Loading τ* values...")
    tau_dict = load_tau_star(TAU_STAR_PATH)
    sigma_map = {
        "encoder_self": "softplus",
        "decoder_self": "sigmoid",
        "decoder_cross": "exp",
    }
    print("Applying τ-opus...")
    apply_tau_opus(model, tau_dict, sigma_map)

    return model.to(DEVICE)


def save_ckpt(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))


# ═══════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Config: bs={BATCH_SIZE}, accum={GRAD_ACCUM}, lr={LR}, epochs={NUM_EPOCHS}")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", cache_dir="/hf_cache",
        language="zh", task="transcribe"
    )

    train_ds = CantoneseASRDataset("/root/dimsum/data/prepared/train.jsonl", processor)
    val_ds = CantoneseASRDataset("/root/dimsum/data/prepared/val.jsonl", processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = setup_model()

    # Smoke test
    print("Smoke test...")
    sample = next(iter(train_loader))
    with torch.no_grad():
        loss = model(input_features=sample["input_features"][:2].to(DEVICE),
                     labels=sample["labels"][:2].to(DEVICE)).loss
        print(f"  Initial loss: {loss.item():.4f}")

    optimizer = AdamW(model.model.decoder.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) * NUM_EPOCHS) // GRAD_ACCUM
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    global_step, best_loss = 0, float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for bidx, batch in enumerate(pbar):
            inp = batch["input_features"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            loss = model(input_features=inp, labels=lbl).loss / GRAD_ACCUM
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
                pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM:.4f}"})
            if global_step > 0 and global_step % 500 == 0:
                save_ckpt(model, os.path.join(OUTPUT_DIR, f"ckpt-{global_step}"))

        avg_loss = total_loss / n_steps
        print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(model, os.path.join(OUTPUT_DIR, "best_model"))

    save_ckpt(model, os.path.join(OUTPUT_DIR, "final_model"))
    print("Done!")

if __name__ == "__main__":
    # Quick self-test on import
    _ = torch.randn(1, 12, 10, 10)
    t = torch.ones(1, 12, 1, 1)
    out = STauOpusMaxStableFn.apply(_, t, "softplus")
    print(f"τ-opus self-test: input {_.shape} -> output {out.shape} (sum={out.sum().item():.2f})")
    main()
