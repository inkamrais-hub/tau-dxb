#!/usr/bin/env python3
"""诊断：加载 erp3 epoch3 checkpoint 跑 10 条 sample 看实际输出"""
import sys, os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

BASE = "/hy-tmp/dimsum"
CKPT = f"{BASE}/outputs/erp3/epoch3/model.pt"
TAU  = f"{BASE}/outputs/erp2_tau_star.json"
sys.path.insert(0, f"{BASE}/scripts")

from patch_whisper_stau import apply_learnable_stau
from patch_whisper_sparse_grad import apply_sparse_grad_wta, set_sparse_grad_k_ratio

device = "cuda"
print("[diag] Loading base model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

print("[diag] Applying patches...")
apply_learnable_stau(model, tau_init=TAU)
apply_sparse_grad_wta(model, encoder_k_ratio=0.5, decoder_k_ratio=0.4, ffn_top_k=20, track_wins=False)
set_sparse_grad_k_ratio(model, 0.5, 0.4)

print("[diag] Loading epoch3 checkpoint...")
state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

print("[diag] Loading dataset + processor...")
ds = load_dataset("leeduckgo/cantonese-life-scenarios-corpus", split="test", cache_dir="/hy-tmp/cache")
proc = WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir="/hy-tmp/cache")
gen_kw = {"language": "zh", "task": "transcribe", "num_beams": 1, "do_sample": False}

from jiwer import cer
errs = []
for i in range(10):
    s = ds[i]
    inp = proc.feature_extractor(s["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        pred_ids = model.generate(inp, **gen_kw)
    hyp = proc.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
    ref = s["text"]
    e = cer(ref, hyp)
    errs.append(e)
    print(f"\n-- #{i+1} CER={e*100:.2f}%")
    print(f"  REF: {repr(ref)}")
    print(f"  HYP: {repr(hyp)}")

print(f"\n[diag] Avg CER= {sum(errs)/len(errs)*100:.2f}%")
