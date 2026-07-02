#!/usr/bin/env python3
"""诊断：加载 erp3 epoch3 checkpoint，跑几条 eval sample 打出来看实际输出"""
import json, sys, os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset
from tqdm import tqdm

BASE      = "/hy-tmp/dimsum"
CKPT_PATH = f"{BASE}/outputs/erp3/epoch3/model.pt"
DEVICE    = "cuda"
N_SAMPLES = 10  # 打 10 条

sys.path.insert(0, f"{BASE}/scripts")
from patch_whisper_stau import apply_learnable_stau
from patch_whisper_sparse_grad import apply_sparse_grad_wta, set_sparse_grad_k_ratio

# ── 加载模型 ──
print("[diag] Loading model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.to(DEVICE)
model.eval()

# ── 加载权重 ──
state = torch.load(CKPT_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"  missing keys: {missing}")
if unexpected:
    print(f"  unexpected keys: {unexpected}")
print(f"  Loaded {CKPT_PATH}")

# ── 应用 patch ──
print("[diag] Applying patches...")
apply_learnable_stau(model, tau_init_mode="from_checkpoint")
apply_sparse_grad_wta(model, 
    encoder_k_ratio=0.5, decoder_k_ratio=0.4, 
    ffn_top_k=20, track_wins=False)
set_sparse_grad_k_ratio(model, 0.5, 0.4)
model.eval()

# ── 加载数据 ──
print("[diag] Loading dataset...")
dataset = load_dataset(
    "leeduckgo/cantonese-life-scenarios-corpus",
    split="test",
    cache_dir="/hy-tmp/cache"
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    cache_dir="/hy-tmp/cache"
)

gen_kwargs = {"language": "zh", "task": "transcribe", "num_beams": 1, "do_sample": False}

# ── 跑 N 条 ──
print(f"[diag] Generating {N_SAMPLES} samples...")
errors = []
for i in range(min(N_SAMPLES, len(dataset))):
    sample = dataset[i]
    inputs = processor.feature_extractor(
        sample["audio"]["array"], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)
    
    with torch.no_grad():
        preds = model.generate(inputs, **gen_kwargs)
    hyp = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)[0]
    ref = sample["text"]
    
    from jiwer import cer
    err = cer(ref, hyp)
    errors.append(err)
    
    print(f"\n── Sample {i+1} ──")
    print(f"  REF: {repr(ref)}")
    print(f"  HYP: {repr(hyp)}")
    print(f"  CER: {err*100:.2f}%")

print(f"\n[diag] Avg CER ({N_SAMPLES} samples): {sum(errors)/len(errors)*100:.2f}%")
print("[diag] Done.")
