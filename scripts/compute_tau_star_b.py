"""
Compute τ* on B model, encoder only. Low-cost version.
- Only encoder self-attention.
- Runs encoder forward directly (no generate / decoder).
- Samples query/key positions only within the valid (non-padded) prefix.
- Newton solver (scalar) instead of SGD on tau.
- 10 runs x 12 samples, median aggregation.

Usage: python scripts/compute_tau_star_b.py
"""
import os, json, random, gc, time
os.environ["HF_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, torch.nn.functional as F
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as mw
from transformers.models.whisper.modeling_whisper import WhisperAttention as WA

CKPT = "/root/dimsum/outputs/abl_b_pure/best_model/model.pt"
OUTPUT = "/root/dimsum/outputs/whisper_small_tau_star_b.json"
VAL_JSONL = "/root/dimsum/data/prepared/val.jsonl"
TRAIN_JSONL = "/root/dimsum/data/prepared/train.jsonl"

N_RUNS = 10
N_SAMPLES = 12
N_QUERY = 8
N_KEY = 256
DEVICE = "cuda"

collector_scores = {}
_orig_forward = None


def patched_eager_attn(module, query, key, value, attention_mask,
                       scaling=None, dropout=0.0, **kwargs):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask

    name = collector_scores.get("_map", {}).get(id(module), "")
    if name:
        s = scores.detach().float().cpu()          # (B, H, Tq, Tk)
        B, H, Tq, Tk = s.shape
        valid_len = collector_scores.get("_valid_len", Tk)
        valid_len = min(valid_len, Tq, Tk)
        if valid_len >= 4:
            # sample query/key positions only within the valid (non-padded) prefix
            nq = min(N_QUERY, valid_len)
            q_idx = torch.randperm(valid_len)[:nq].sort().values
            nk = min(N_KEY, valid_len)
            start = random.randint(0, valid_len - nk)
            k_idx = torch.arange(start, start + nk)
            collector_scores.setdefault(name, []).append(
                s[:, :, q_idx, :][:, :, :, k_idx]
            )

    attn_weights = F.softmax(scores.float(), dim=-1).to(value.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    return torch.matmul(attn_weights, value).transpose(1, 2).contiguous(), attn_weights


def install(modules):
    global _orig_forward
    collector_scores.clear()
    collector_scores["_map"] = {id(mod): name for name, mod in modules}
    _orig_forward = mw.eager_attention_forward
    mw.eager_attention_forward = patched_eager_attn


def uninstall():
    global _orig_forward
    if _orig_forward is not None:
        mw.eager_attention_forward = _orig_forward
        _orig_forward = None


def solve_tau_star(scores, n_iter=30):
    """
    Minimise KL( softmax(scores) || softplus(scores)^tau / Z ).
    Scalar Newton on tau; scores can be any shape, last dim = keys.
    """
    ref = F.softmax(scores, dim=-1).float()
    log_sigma = F.softplus(scores).clamp(min=1e-8).log().float()
    E_ref = (ref * log_sigma).sum(dim=-1).mean()

    tau = torch.tensor(1.0, device=scores.device, dtype=torch.float32)
    for _ in range(n_iter):
        log_p = tau * log_sigma
        log_Z = torch.logsumexp(log_p, dim=-1, keepdim=True)
        p = (log_p - log_Z).exp()
        E_p = (p * log_sigma).sum(dim=-1).mean()
        var_p = (p * log_sigma * log_sigma).sum(dim=-1).mean() - E_p * E_p
        g = E_p - E_ref
        tau = tau - g / var_p.clamp(min=1e-8)
        tau = tau.clamp(0.1, 10.0)
    return tau.item()


def load_audio(ap):
    if not os.path.exists(ap):
        return None
    try:
        audio, sr = sf.read(ap)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.ndim > 1:
            audio = audio.mean(-1)
        return audio.astype(np.float32)
    except Exception as e:
        return None


def main():
    random.seed(42)
    torch.manual_seed(42)
    print(f"Device: {DEVICE}")
    print(f"Config: runs={N_RUNS}, samples/run={N_SAMPLES}, "
          f"query={N_QUERY}, key={N_KEY}")

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        attn_implementation="eager",
        cache_dir="/hf_cache",
    )
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small",
        cache_dir="/hf_cache",
        language="zh",
        task="transcribe",
    )

    with open(VAL_JSONL) as f:
        pool = [json.loads(l) for l in f if l.strip()]
    with open(TRAIN_JSONL) as f:
        train = [json.loads(l) for l in f if l.strip()]
    if len(train) > 300:
        train = random.sample(train, 300)
    pool.extend(train)
    print(f"Pool size: {len(pool)}")

    enc_modules = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, WA) and not getattr(m, "is_decoder", False)
    ]
    print(f"Encoder attention modules: {len(enc_modules)}")

    install(enc_modules)
    all_run_taus = []

    for run in range(N_RUNS):
        t0 = time.time()
        print(f"\n=== Run {run + 1}/{N_RUNS} ===")
        collector_scores.clear()
        collector_scores["_map"] = {id(mod): name for name, mod in enc_modules}

        subset = random.sample(pool, min(N_SAMPLES, len(pool)))
        for item in tqdm(subset, desc=f"  forward"):
            audio = load_audio(item["audio_path"])
            if audio is None:
                continue

            inputs = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(DEVICE)

            # Whisper encoder requires exactly 3000 frames; we keep full length
            # but only sample attention positions within the non-padded region.
            T = inputs.shape[-1]
            valid_frames = min(T, max(1, int(np.ceil(len(audio) / 160))))
            collector_scores["_valid_len"] = valid_frames

            with torch.no_grad():
                model.model.encoder(inputs)

        # per-head tau* within this run
        run_taus = {}
        for name, mod in enc_modules:
            scores_list = collector_scores.get(name, [])
            if len(scores_list) < 3:
                run_taus[name] = [1.0] * mod.num_heads
                continue

            head_taus = []
            for h in range(mod.num_heads):
                taus = []
                for s in scores_list:  # (1, H, nq, nk)
                    hs = s[0, h].to(DEVICE)  # (nq, nk)
                    if hs.shape[-1] < 4:
                        continue
                    taus.append(solve_tau_star(hs))
                head_taus.append(round(float(np.median(taus)), 4) if len(taus) >= 3 else 1.0)
            run_taus[name] = head_taus
            print(f"  {name.split('.')[-1]:30s} τ*={np.mean(head_taus):.2f}")

        all_run_taus.append(run_taus)
        print(f"  run time: {time.time() - t0:.1f}s")
        collector_scores.pop("_valid_len", None)
        gc.collect()
        torch.cuda.empty_cache()

    uninstall()

    # aggregate across runs by median
    print(f"\n{'='*60}\nAggregating {N_RUNS} runs ...")
    results = []
    for name, mod in enc_modules:
        per_run = [run_taus[name] for run_taus in all_run_taus]
        medians = np.median(per_run, axis=0).tolist()
        results.append({
            "module_name": name,
            "num_heads": mod.num_heads,
            "tau_per_head": [round(t, 4) for t in medians],
            "tau_mean": round(float(np.mean(medians)), 4),
        })
        print(f"  {name.split('.')[-1]:30s} τ*_med={np.mean(medians):.2f}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    all_t = [t for r in results for t in r["tau_per_head"]]
    print(f"\nSaved: {OUTPUT}")
    print(f"Encoder τ* overall: mean={np.mean(all_t):.2f} ± {np.std(all_t):.2f}")


if __name__ == "__main__":
    main()
