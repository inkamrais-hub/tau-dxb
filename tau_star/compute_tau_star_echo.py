"""
compute_tau_star_echo.py — echo-opus-1 专用 τ* 求解器

核心改进（相对 compute_whisper_tau_star.py）：
  1. per-type σ 函数：encoder_self=softplus, decoder_self=sigmoid, decoder_cross=exp
  2. 用真实粤语语音数据（非随机噪声），确保 τ* 反映实际声学分布
  3. KL 散度验证：比较 τ*-σ^τ 归一化 vs 原始 softmax 的分布差异
  4. 输出 JSON 直接兼容 apply_learnable_stau() 的 tau_init 参数

用法：
  python scripts/compute_tau_star_echo.py \
    --data_jsonl F:/τ/点心杯/data/life-scenarios/train.jsonl \
    --model openai/whisper-small \
    --n_samples 200 \
    --output outputs/echo-opus-1/tau_star.json
"""
import os, sys, json, math, random, argparse
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("HF_HOME", r"F:\hf_cache")
os.environ.setdefault("TORCH_HOME", r"F:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TORCHAUDIO_USE_EXTENSION", "0")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.11/dist-packages/torchaudio/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# Import tau_star_opus — works both locally (F:\τ\τopus\) and on server (scripts/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
# Also try local dev path
_local_topus = r"F:\τ\τopus"
if os.path.isdir(_local_topus) and _local_topus not in sys.path:
    sys.path.insert(0, _local_topus)
from tau_star_opus import TauStarOpus

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module
import soundfile as sf
import librosa

from tau_star_solver import solve_tau_star

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_RATE = 16000

# Per-type σ → TauStarOpus constructor
_SIGMA_ESTIMATORS = {
    "encoder_self":  ("softplus", TauStarOpus.softplus_closed),
    "decoder_self":  ("sigmoid",  TauStarOpus.sigmoid_closed),
    "decoder_cross": ("exp",      TauStarOpus.exp_closed),
}

# Global estimators cache
_GLOBAL_ESTIMATORS = {
    "encoder_self": TauStarOpus.softplus_closed(),
    "decoder_self": TauStarOpus.sigmoid_closed(),
    "decoder_cross": TauStarOpus.exp_closed(),
}

AGG_BUFFER = {}  # key -> list of (taus, kls) tuples


def patched_eager_attention_forward(module, query, key, value, attention_mask,
                                    scaling=None, dropout=0.0, **kwargs):
    """捕获 pre-softmax attention scores，list 收集避免 O(N²) cat。"""
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    is_decoder = getattr(module, "is_decoder", False)
    cross = (key.shape[2] != query.shape[2])

    scores_cpu = attn_weights.detach().to("cpu", dtype=torch.float32)

    layer_idx = getattr(module, "layer_idx", -1)
    if layer_idx is None:
        layer_idx = -1
    if is_decoder and cross:
        kind = "decoder_cross"
    elif is_decoder:
        kind = "decoder_self"
    else:
        kind = "encoder_self"
    key = f"{kind}|{layer_idx}"

    # Instantly compute tau* and KL to avoid OOM
    estimator = _GLOBAL_ESTIMATORS[kind]
    sigma_name = _SIGMA_ESTIMATORS[kind][0]
    taus, kls = estimate_tau_per_head(scores_cpu, estimator, sigma_name)
    
    # Collect scalars to list — O(1) memory, not O(N) tensors
    if key not in AGG_BUFFER:
        AGG_BUFFER[key] = []
    AGG_BUFFER[key].append((taus, kls))

    attn_weights_sm = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights_sm = F.dropout(attn_weights_sm, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights_sm.to(value.dtype), value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights_sm


def _classify(item):
    if not item["is_decoder"]:
        return "encoder_self"
    if item.get("cross_attention", False):
        return "decoder_cross"
    return "decoder_self"


def estimate_tau_per_head(scores_tensor, estimator, sigma_name="softplus"):
    """
    scores_tensor: [B, H, Lq, Lk]
    返回 (tau_list[H], kl_list[H]) — 高精度 Newton 求解的 τ* 和对应 KL
    """
    B, H, Lq, Lk = scores_tensor.shape
    taus, kls = [], []
    for h in range(H):
        s = scores_tensor[:, h, :, :].reshape(-1, Lk)  # [B*Lq, Lk]
        
        # 使用高精度 Newton solver 求解，迭代 100 次确保收敛
        tau_tensor, kl_tensor = solve_tau_star(s, sigma=sigma_name, n_iter=100)
        
        # tau_tensor 形状也是 [B*Lq]，我们需要求平均
        tau_mean = float(tau_tensor.mean())
        kl_mean = float(kl_tensor.mean())

        taus.append(round(tau_mean, 4))
        kls.append(round(kl_mean, 6))

    return taus, kls


def load_audio_loader(jsonl_path, n_samples):
    """从 JSONL 加载前 n_samples 条音频路径。"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    random.shuffle(data)
    paths = []
    for item in data[:n_samples]:
        ap = item.get("audio_path", item.get("audio", ""))
        if ap and os.path.exists(ap):
            paths.append(ap)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_jsonl", type=str,
                        default=r"F:\τ\点心杯\data\life-scenarios\train.jsonl")
    parser.add_argument("--model", type=str, default="openai/whisper-small")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of real audio samples to use")
    parser.add_argument("--output", type=str,
                        default=r"F:\τ\点心杯\outputs\echo-opus-1\tau_star.json")
    parser.add_argument("--max_audio_sec", type=float, default=8.0,
                        help="Truncate audio to this many seconds")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[τ* echo] Device: {DEVICE}")
    print(f"[τ* echo] Data: {args.data_jsonl}, n_samples={args.n_samples}")
    print(f"[τ* echo] Output: {args.output}")

    # 1. Load audio paths
    audio_paths = load_audio_loader(args.data_jsonl, args.n_samples)
    print(f"[τ* echo] Loaded {len(audio_paths)} audio files")

    # 2. Load processor + model
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, attn_implementation="eager",
    ).to(DEVICE).eval()

    # 3. Set layer_idx on all WhisperAttention modules
    for n, m in model.named_modules():
        if isinstance(m, whisper_module.WhisperAttention):
            lidx = -1
            if "encoder.layers." in n:
                lidx = int(n.split("encoder.layers.")[1].split(".")[0])
            elif "decoder.layers." in n:
                lidx = int(n.split("decoder.layers.")[1].split(".")[0])
            m.layer_idx = lidx

    # 4. Patch attention
    original_forward = whisper_module.eager_attention_forward
    whisper_module.eager_attention_forward = patched_eager_attention_forward

    max_frames = int(args.max_audio_sec * 100)  # ~800 mel frames for 8 sec

    try:
        # 4. Run forward passes on real audio
        print(f"[τ* echo] Running {len(audio_paths)} forward passes ...")
        for i, ap in enumerate(audio_paths):
            if i % 50 == 0:
                print(f"  {i}/{len(audio_paths)}")
            try:
                audio, sr = sf.read(ap)
                if sr != SAMPLING_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
                if audio.ndim > 1:
                    audio = audio.mean(-1)
                audio = audio.astype("float32")
            except Exception:
                continue

            feat = processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE,
                                               return_tensors="pt").input_features
            # Pad/truncate to 3000 mel frames (Whisper hard requirement)
            feat = feat.to(DEVICE)
            if feat.shape[-1] < 3000:
                feat = F.pad(feat, (0, 3000 - feat.shape[-1]))
            feat = feat[:, :, :3000]
            decoder_ids = torch.randint(0, model.config.vocab_size, (1, 10)).to(DEVICE)

            with torch.no_grad():
                _ = model(input_features=feat, decoder_input_ids=decoder_ids)

        print(f"[τ* echo] Captured attention scores and computed tau* on the fly for {sum(len(v) for v in AGG_BUFFER.values())} invocations")

        # 6. Aggregate estimated τ* per head
        results = []
        kl_summary = {}
        for kind in ["encoder_self", "decoder_self", "decoder_cross"]:
            sigma_name = _SIGMA_ESTIMATORS[kind][0]
            kl_summary[kind] = {"kl_per_head": [], "tau_list": []}

            for key, stats_list in sorted(AGG_BUFFER.items()):
                k, lidx = key.split("|")
                if k != kind:
                    continue
                
                # stats_list is a list of (taus, kls) tuples where taus has length H
                if not stats_list:
                    continue
                    
                H = len(stats_list[0][0])
                avg_taus = [0.0] * H
                avg_kls = [0.0] * H
                
                for taus, kls in stats_list:
                    for h in range(H):
                        avg_taus[h] += taus[h]
                        avg_kls[h] += kls[h]
                
                n = len(stats_list)
                avg_taus = [round(v / n, 4) for v in avg_taus]
                avg_kls = [round(v / n, 6) for v in avg_kls]
                
                results.append({
                    "attention_type": kind,
                    "sigma": sigma_name,
                    "layer_idx": int(lidx),
                    "tau_per_head": avg_taus,
                    "tau_mean": round(sum(avg_taus)/len(avg_taus), 4),
                    "tau_min": round(min(avg_taus), 4),
                    "tau_max": round(max(avg_taus), 4),
                    "kl_per_head": avg_kls,
                    "kl_mean": round(sum(avg_kls)/len(avg_kls), 6),
                })
                kl_summary[kind]["kl_per_head"].extend(avg_kls)
                kl_summary[kind]["tau_list"].extend(avg_taus)

        # 7. Summary
        print("\n=== τ* 估计结果 ===")
        for kind in ["encoder_self", "decoder_self", "decoder_cross"]:
            items = [r for r in results if r["attention_type"] == kind]
            if not items:
                continue
            all_taus = [t for r in items for t in r["tau_per_head"]]
            all_kls = [k for r in items for k in r["kl_per_head"]]
            print(f"\n  [{kind}] sigma={_SIGMA_ESTIMATORS[kind][0]}")
            print(f"    τ:  mean={sum(all_taus)/len(all_taus):.3f}, "
                  f"min={min(all_taus):.3f}, max={max(all_taus):.3f}")
            print(f"    KL: mean={sum(all_kls)/len(all_kls):.6f}, "
                  f"max={max(all_kls):.6f}, layers={len(items)}")
            # KL quality assessment
            kl_mean = sum(all_kls)/len(all_kls)
            if kl_mean < 0.01:
                quality = "EXCELLENT (τ* nearly perfect match)"
            elif kl_mean < 0.05:
                quality = "GOOD (minor distribution mismatch)"
            elif kl_mean < 0.1:
                quality = "FAIR (noticeable but acceptable)"
            else:
                quality = "POOR (significant mismatch, τ* may need adjustment)"
            print(f"    Quality: {quality}")

        # 8. Save
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[τ* echo] Saved to {args.output}")

        # Also save a kl_report
        kl_path = os.path.join(out_dir, "kl_report.json")
        with open(kl_path, "w", encoding="utf-8") as f:
            json.dump({
                "description": "KL(p_softmax || p_tau_star) per attention type",
                "per_type": {
                    k: {
                        "sigma": _SIGMA_ESTIMATORS[k][0],
                        "kl_mean": round(sum(v["kl_per_head"])/max(1, len(v["kl_per_head"])), 6),
                        "kl_max": round(max(v["kl_per_head"]), 6) if v["kl_per_head"] else 0,
                    }
                    for k, v in kl_summary.items()
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"[τ* echo] KL report: {kl_path}")

    finally:
        whisper_module.eager_attention_forward = original_forward


if __name__ == "__main__":
    main()
