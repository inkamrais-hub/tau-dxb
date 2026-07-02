"""
计算 Whisper-small 各 attention head 的 τ* 分布。
用法：python scripts/compute_whisper_tau_star.py
"""
import os
import sys
import json
import math
from pathlib import Path

# 缓存到 F 盘，避免占 C 盘
os.environ.setdefault("HF_HOME", r"F:\hf_cache")
os.environ.setdefault("TORCH_HOME", r"F:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 引入 τ-opus 求解器
sys.path.insert(0, r"F:\τ\τopus")
from tau_star_opus import TauStarOpus

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32 if DEVICE.type == "cpu" else torch.float16

# 用于收集 pre-softmax attention scores
SCORE_BUFFER = []


def patched_eager_attention_forward(module, query, key, value, attention_mask,
                                    scaling=None, dropout=0.0, **kwargs):
    """替换 whisper 的 eager attention，捕获 pre-softmax scores 后走原路。"""
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # 捕获：只保留 [B, H, Lq, Lk] 的 attention score
    SCORE_BUFFER.append({
        "layer_idx": getattr(module, "layer_idx", -1),
        "is_decoder": getattr(module, "is_decoder", False),
        "cross_attention": (key.shape[2] != query.shape[2]),  # 简单判断 cross vs self
        "scores": attn_weights.detach().to("cpu", dtype=torch.float32),
    })

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights.to(value.dtype), value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def estimate_tau_per_head(scores_tensor):
    """
    scores_tensor: [B, H, Lq, Lk]
    返回 [H] 的 τ* 列表（对每个 head，把 B*Lq 条 query 拼起来估计）
    """
    est = TauStarOpus.softplus_closed()
    B, H, Lq, Lk = scores_tensor.shape
    taus = []
    for h in range(H):
        # 拼成 [B*Lq, Lk]
        s = scores_tensor[:, h, :, :].reshape(-1, Lk)
        tau, _ = est._estimate(s)
        taus.append(float(tau))
    return taus


def main():
    print(f"Using device: {DEVICE}, dtype: {DTYPE}")

    # 把 whisper 的 eager attention 换成捕获版
    original_forward = whisper_module.eager_attention_forward
    whisper_module.eager_attention_forward = patched_eager_attention_forward

    try:
        print("Loading openai/whisper-small ...")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small",
            attn_implementation="eager",
            torch_dtype=DTYPE,
        ).to(DEVICE).eval()

        # Whisper encoder 要求 mel 长度 3000（30 秒），用随机噪声做标定即可
        input_features = torch.randn(1, 80, 3000).to(DEVICE, dtype=DTYPE)
        # decoder 输入：batch=1, length=10 的随机 token id
        decoder_input_ids = torch.randint(0, model.config.vocab_size, (1, 10)).to(DEVICE)

        print("Running one forward pass to capture attention scores ...")
        with torch.no_grad():
            _ = model(input_features=input_features, decoder_input_ids=decoder_input_ids)

        print(f"Captured {len(SCORE_BUFFER)} attention score tensors.")

        # 按层/类型聚合
        results = []
        for item in SCORE_BUFFER:
            taus = estimate_tau_per_head(item["scores"])
            results.append({
                "layer_idx": item["layer_idx"],
                "is_decoder": item["is_decoder"],
                "cross_attention": item["cross_attention"],
                "tau_per_head": taus,
                "tau_mean": float(sum(taus)) / len(taus),
                "tau_std": float(torch.tensor(taus).std().item()),
                "tau_min": min(taus),
                "tau_max": max(taus),
            })

        # 保存结果
        out_dir = Path(r"F:\τ\点心杯\outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "whisper_small_tau_star.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 打印汇总
        enc_layers = [r for r in results if not r["is_decoder"]]
        dec_self = [r for r in results if r["is_decoder"] and not r["cross_attention"]]
        dec_cross = [r for r in results if r["is_decoder"] and r["cross_attention"]]

        print("\n=== τ* 分布汇总 (softplus closed form) ===")
        for name, group in [("encoder self", enc_layers),
                            ("decoder self", dec_self),
                            ("decoder cross", dec_cross)]:
            if not group:
                continue
            all_taus = [t for r in group for t in r["tau_per_head"]]
            print(f"{name}: n={len(all_taus)}, mean={sum(all_taus)/len(all_taus):.3f}, "
                  f"std={torch.tensor(all_taus).std().item():.3f}, "
                  f"min={min(all_taus):.3f}, max={max(all_taus):.3f}")
            # 每头平均
            per_head_avg = [sum(r["tau_per_head"][h] for r in group) / len(group)
                            for h in range(12)]
            print(f"  per-head avg across {len(group)} layers: "
                  f"[{', '.join(f'{x:.2f}' for x in per_head_avg)}]")

        print(f"\nDetailed results saved to: {out_path}")

    finally:
        # 恢复原始 forward，避免污染全局
        whisper_module.eager_attention_forward = original_forward


if __name__ == "__main__":
    main()
