"""
Whisper Base-XL inference → submission.jsonl + CER evaluation.
Usage: python3.11 scripts/run_inference.py
"""
import os, sys, json
os.environ["HF_HOME"] = "/hf_cache"
os.environ["TORCH_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module
import soundfile as sf
from tqdm import tqdm

# ── Config ──
CKPT_DIR = "/root/dimsum/outputs/integrated/best_model"
TEST_JSONL = "/root/dimsum/data/test.jsonl"
OUTPUT_JSONL = "/root/dimsum/outputs/integrated/submission.jsonl"
EVALUATOR = "/root/dimsum/data/evaluator.py"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLING_RATE = 16000

# ── τ-opus (inlined for inference) ──
CLAMP_MIN = 1e-8
def _sigma_softplus(x): return F.softplus(x).clamp(min=CLAMP_MIN)
def _sigma_sigmoid(x):  return torch.sigmoid(x).clamp(min=CLAMP_MIN)
def _sigma_exp(x):      return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)
_SIGMA = {"softplus": _sigma_softplus, "sigmoid": _sigma_sigmoid, "exp": _sigma_exp}
def _sigma_prime_softplus(x): return torch.sigmoid(x)
def _sigma_prime_sigmoid(x):  s = torch.sigmoid(x); return s * (1 - s)
def _sigma_prime_exp(x):      return torch.exp(x.clamp(max=15))
_SIGMA_PRIME = {"softplus": _sigma_prime_softplus, "sigmoid": _sigma_prime_sigmoid, "exp": _sigma_prime_exp}

class STauOpusMaxStableFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, tau, sigma_name):
        sigma_fn = _SIGMA[sigma_name]
        x_stable = scores - scores.max(dim=-1, keepdim=True).values
        sigma_val = sigma_fn(x_stable)
        q = sigma_val.clamp(min=CLAMP_MIN).pow(tau)
        sum_q = q.sum(dim=-1, keepdim=True).clamp(min=CLAMP_MIN)
        attn = q / sum_q
        ctx.save_for_backward(scores, x_stable, sigma_val, attn, sum_q)
        ctx.tau = tau; ctx.sigma_name = sigma_name
        return attn
    @staticmethod
    def backward(ctx, dO):
        scores, x_stable, sigma_val, attn, sum_q = ctx.saved_tensors
        tau, sigma_name = ctx.tau, ctx.sigma_name
        sigma_prime_fn = _SIGMA_PRIME.get(sigma_name, lambda x: torch.ones_like(x))
        sigma_tau_m1 = sigma_val.clamp(min=CLAMP_MIN).pow(tau - 1.0)
        sigma_p = sigma_prime_fn(x_stable)
        A = tau * sigma_tau_m1 * sigma_p
        S = sum_q; q = sigma_val.clamp(min=CLAMP_MIN).pow(tau)
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

def _make_stau_forward():
    def stau_eager_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if scaling is None: scaling = query.size(-1) ** -0.5
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None: scores = scores + attention_mask
        tau = getattr(module, "_stau_tau", torch.tensor(1.0, device=scores.device)).to(scores.device)
        sigma = getattr(module, "_stau_sigma", "softplus")
        attn_weights = STauOpusMaxStableFn.apply(scores.float(), tau, sigma)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights.to(value.dtype), value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights.to(value.dtype)
    return stau_eager_forward

def restore_tau_opus(model):
    """Re-apply τ-opus to a model that was saved with _stau_tau buffers."""
    whisper_module.eager_attention_forward = _make_stau_forward()
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "WhisperAttention" and hasattr(module, "_stau_tau"):
            count += 1
    print(f"  τ-opus restored on {count} attention modules")


# ── Main ──
def test_inference():
    """Load test data, remap audio paths, run inference, evaluate."""
    # Load test data
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"Test samples: {len(test_data)}")

    # Build filename→path mapping
    import subprocess
    result = subprocess.run(
        ["find", "/root/dimsum/data/audio_data", "-name", "*.wav"],
        capture_output=True, text=True, timeout=60
    )
    all_wavs = result.stdout.strip().split("\n")
    path_map = {}
    for wav in all_wavs:
        fname = os.path.basename(wav)
        if fname not in path_map:  # first match wins
            path_map[fname] = wav
    print(f"Audio files found: {len(path_map)}")

    # Remap test data
    remapped = []
    missing = 0
    for item in test_data:
        fname = os.path.basename(item["audio_path"])
        if fname in path_map:
            item["audio_path"] = path_map[fname]
            remapped.append(item)
        else:
            missing += 1
    if missing:
        print(f"WARNING: {missing}/{len(test_data)} audio files not found")
    print(f"Remapped: {len(remapped)} test samples")

    # Load processor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", cache_dir="/hf_cache",
        language="zh", task="transcribe"
    )

    # Load best model checkpoint
    state_dict_path = os.path.join(CKPT_DIR, "model.pt")
    if not os.path.exists(state_dict_path):
        files = os.listdir(CKPT_DIR)
        pt_files = [f for f in files if f.endswith((".pt", ".pth", ".ckpt", ".safetensors"))]
        if pt_files:
            state_dict_path = os.path.join(CKPT_DIR, pt_files[0])
        else:
            raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")
    print(f"Loading: {state_dict_path}")

    # Load model fresh
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", attn_implementation="eager", cache_dir="/hf_cache"
    )

    # Load state dict (missing _stau_tau buffers expected — we'll restore them after)
    state_dict = torch.load(state_dict_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    # Restore τ-opus: register _stau_tau buffers from loaded state_dict
    count = 0
    sigma_map = {"encoder_self": "softplus", "decoder_self": "sigmoid", "decoder_cross": "exp"}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "WhisperAttention":
            continue
        tau_key = name + "._stau_tau"
        if tau_key not in state_dict:
            continue
        module.register_buffer("_stau_tau", state_dict[tau_key].to(DEVICE))
        kind = ("decoder_cross" if ("encoder_attn" in name or "cross" in name.lower()) else
                ("decoder_self" if getattr(module, "is_decoder", False) else "encoder_self"))
        module._stau_sigma = sigma_map.get(kind, "softplus")
        count += 1
    restore_tau_opus(model)
    print(f"τ-opus: {count} attention modules restored")
    model.to(DEVICE)
    model.eval()
    print(f"Model on {DEVICE}")

    # Run inference
    results = []
    gen_kwargs = {
        "language": "zh",
        "task": "transcribe",
        "num_beams": 5,
        "temperature": 0.0,
        "do_sample": False,
    }

    for item in tqdm(test_data, desc="Inferring"):
        audio, sr = sf.read(item["audio_path"])
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        if audio.ndim > 1:
            audio = audio.mean(-1)

        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features.to(DEVICE)

        with torch.no_grad():
            predicted_ids = model.generate(inputs, **gen_kwargs)
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

        results.append({
            "key": item.get("key", os.path.splitext(os.path.basename(item["audio_path"]))[0]),
            "predict": transcription,
            "ground_truth": item.get("text", ""),
        })

    # Save submission
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {OUTPUT_JSONL} ({len(results)} predictions)")

    # Run official evaluator
    sys.path.insert(0, os.path.dirname(EVALUATOR))
    try:
        from evaluator import evaluate
        cer, confusion, recall = evaluate(OUTPUT_JSONL)
        print(f"\n{'='*50}")
        print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
        if confusion:
            print(f"Confusion pairs: {len(confusion)}")
            for pair, cnt in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
                print(f"  {pair}: {cnt}")
        if recall:
            print(f"Tone word recall: {recall}")
    except ImportError as e:
        print(f"Skipping evaluator (not available): {e}")

if __name__ == "__main__":
    test_inference()
