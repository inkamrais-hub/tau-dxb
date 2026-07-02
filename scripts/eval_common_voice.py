"""
Evaluate Base-XL on Common Voice 17 yue test set.
Usage: python scripts/eval_common_voice.py
"""
import os, sys, json, subprocess, tempfile
os.environ["HF_HOME"] = "/hf_cache"
os.environ["TORCH_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

CKPT = "/root/dimsum/outputs/integrated/best_model/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "/root/dimsum/outputs/cv_test_results.json"

# τ-opus inlined (same as run_inference.py)
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

def char_error_rate(refs, hyps):
    """CER computation matching the official evaluator's logic."""
    import Levenshtein
    total_ed, total_chars = 0, 0
    for r, h in zip(refs, hyps):
        total_ed += Levenshtein.distance(r, h)
        total_chars += len(r)
    return total_ed / total_chars if total_chars > 0 else 1.0

def main():
    print(f"Device: {DEVICE}")
    print("Loading Common Voice 17 yue (test split)...")
    ds = load_dataset("kaschung4/common_voice_17_yue_pseudo_labelled", split="test")
    print(f"Test samples: {len(ds)}")

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", cache_dir="/hf_cache",
        language="zh", task="transcribe"
    )

    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", attn_implementation="eager", cache_dir="/hf_cache"
    )
    state_dict = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    sigma_map = {"encoder_self": "softplus", "decoder_self": "sigmoid", "decoder_cross": "exp"}
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "WhisperAttention":
            continue
        tau_key = name + "._stau_tau"
        if tau_key not in state_dict:
            continue
        module.register_buffer("_stau_tau", state_dict[tau_key].to(DEVICE))
        kind = ("decoder_cross" if ("encoder_attn" in name) else
                ("decoder_self" if getattr(module, "is_decoder", False) else "encoder_self"))
        module._stau_sigma = sigma_map.get(kind, "softplus")
        count += 1
    whisper_module.eager_attention_forward = _make_stau_forward()
    print(f"τ-opus: {count} modules restored")
    model.to(DEVICE)
    model.eval()

    gen_kwargs = {
        "language": "zh", "task": "transcribe",
        "num_beams": 5, "temperature": 0.0, "do_sample": False,
    }

    refs, hyps = [], []
    for i, item in enumerate(tqdm(ds, desc="CV Eval")):
        # Use path-based access to avoid torchcodec
        audio_path = item["audio"]["path"]
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.ndim > 1:
            audio = audio.mean(-1)

        inputs = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)

        with torch.no_grad():
            pred_ids = model.generate(inputs, **gen_kwargs)
            text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]

        ref_text = item.get("sentence", item.get("text", ""))
        refs.append(ref_text)
        hyps.append(text)

        if (i + 1) % 100 == 0:
            interim_cer = char_error_rate(refs, hyps)
            print(f"  [{i+1}/{len(ds)}] Interim CER: {interim_cer:.4f}")

    cer = char_error_rate(refs, hyps)
    sent_acc = sum(1 for r, h in zip(refs, hyps) if r == h) / len(refs)

    results = {
        "dataset": "Common Voice 17 yue (test)",
        "num_samples": len(refs),
        "cer": cer,
        "sentence_accuracy": sent_acc,
        "correct": sum(1 for r, h in zip(refs, hyps) if r == h),
        "total": len(refs),
    }
    print(f"\n{'='*50}")
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    print(f"Sentence Acc: {sent_acc:.4f} ({sent_acc*100:.2f}%)")
    print(f"Correct: {results['correct']}/{results['total']}")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
