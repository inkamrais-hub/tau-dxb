"""
Full-model τ* estimation on the base-iop checkpoint.
Collects encoder self-attn, decoder self-attn, and decoder cross-attn scores,
then uses the standalone tau_star_solver operator.
"""
import os, sys, json, gc, random
os.environ["HF_HOME"] = "/hy-tmp/hf_cache"
os.environ["TORCH_HOME"] = "/hy-tmp/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.modeling_whisper import eager_attention_forward as _orig_eager
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from tau_star_solver import TauStarEstimator

CKPT_PATH = "/hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = "/hy-tmp/dimsum/outputs/base-erp1/final_model/model.pt"

OUTPUT_PATH = "/hy-tmp/dimsum/outputs/erp2_tau_star.json"
DATA_PATH = "/hy-tmp/dimsum/data/prepared/val.jsonl"

N_RUNS = 5
SAMPLES_PER_RUN = 20
MAX_AUDIO_SEC = 30.0
MAX_NEW_TOKENS = 32
SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# module name -> sigma type (first match wins, order matters)
SIGMA_MAP = {
    "model.encoder.layers.": "softplus",
    "encoder_attn": "exp",
    "model.decoder.layers.": "sigmoid",
}

_estimator: TauStarEstimator | None = None
_id2name: dict = {}
_collecting = False


def _is_encoder_self(name: str) -> bool:
    return "model.encoder.layers." in name and "self_attn" in name


def _is_decoder_self(name: str) -> bool:
    return "model.decoder.layers." in name and "self_attn" in name


def _is_decoder_cross(name: str) -> bool:
    return "model.decoder.layers." in name and "encoder_attn" in name


def _sample_scores(name: str, scores: torch.Tensor) -> torch.Tensor:
    """
    scores: (B, H, nq, nk)
    Downsample query/key positions to keep memory tiny.
    """
    b, h, nq, nk = scores.shape
    if _is_encoder_self(name):
        nq_sample = min(8, nq)
        nk_sample = min(256, nk)
        q_idx = torch.randperm(nq)[:nq_sample]
        k_idx = torch.randperm(nk)[:nk_sample]
        return scores[:, :, q_idx][:, :, :, k_idx]
    if _is_decoder_self(name):
        # keep all keys (sequence is short)
        return scores
    if _is_decoder_cross(name):
        nk_sample = min(256, nk)
        k_idx = torch.randperm(nk)[:nk_sample]
        return scores[:, :, :, k_idx]
    # default
    return scores


def _patched_eager(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask

    if _collecting:
        name = _id2name.get(id(module), "")
        if name and _estimator is not None:
            sampled = _sample_scores(name, scores.detach())
            _estimator.collect(name, sampled)

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def load_val_samples():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def load_audio(item):
    audio, sr = sf.read(item["audio_path"])
    if sr != SAMPLING_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    return audio


def main():
    print(f"[tau* iop] Device: {DEVICE}")
    print(f"[tau* iop] checkpoint: {CKPT_PATH}")

    MODEL_PATH = "/hy-tmp/whisper-small-local"
    processor = WhisperProcessor.from_pretrained(
        MODEL_PATH, cache_dir="/hy-tmp/hf_cache",
        language="zh", task="transcribe", local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        cache_dir="/hy-tmp/hf_cache",
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"[tau* iop] loaded checkpoint")
    else:
        print(f"[tau* iop] WARNING: checkpoint not found, using pretrained base")
    model.to(DEVICE)
    model.eval()

    # build id -> name map
    global _id2name
    for n, m in model.named_modules():
        if "self_attn" in n or "encoder_attn" in n:
            _id2name[id(m)] = n

    # patch globally
    import transformers.models.whisper.modeling_whisper as whisper_module
    whisper_module.eager_attention_forward = _patched_eager

    samples = load_val_samples()

    per_run_results = []
    for run in range(N_RUNS):
        seed = 42 + run
        random.seed(seed)
        torch.manual_seed(seed)

        run_samples = random.sample(samples, SAMPLES_PER_RUN)
        global _estimator, _collecting
        _estimator = TauStarEstimator(
            n_iter=30,
            tau_init=2.0,
            tau_min=0.1,
            tau_max=10.0,
            sigma_map=SIGMA_MAP,
        )
        _collecting = True

        for item in tqdm(run_samples, desc=f"Run {run+1}/{N_RUNS}"):
            audio = load_audio(item)
            # truncate overly long audio
            if len(audio) > int(MAX_AUDIO_SEC * SAMPLING_RATE):
                audio = audio[: int(MAX_AUDIO_SEC * SAMPLING_RATE)]
            inputs = processor.feature_extractor(
                audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
            )
            input_features = inputs.input_features.to(DEVICE)

            with torch.no_grad():
                # encoder forward
                _ = model.model.encoder(input_features)
                # decoder generate
                model.generate(
                    input_features,
                    language="zh",
                    task="transcribe",
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0,
                )

        _collecting = False
        results = _estimator.aggregate(device=DEVICE)
        per_run_results.append([r.to_dict() for r in results])

        # free memory
        _estimator = None
        gc.collect()
        torch.cuda.empty_cache()

    # median tau across runs
    medians = []
    for idx, mod in enumerate(per_run_results[0]):
        name = mod["module_name"]
        heads = mod["num_heads"]
        tau_med = []
        kl_mean = []
        for h in range(heads):
            taus = sorted([per_run_results[r][idx]["tau_per_head"][h] for r in range(N_RUNS)])
            kls = [per_run_results[r][idx]["kl_per_head"][h] for r in range(N_RUNS)]
            tau_med.append(round(taus[len(taus)//2], 4))
            kl_mean.append(round(sum(kls)/len(kls), 6))
        medians.append({
            "module_name": name,
            "num_heads": heads,
            "tau_per_head": tau_med,
            "kl_per_head": kl_mean,
            "tau_mean": round(sum(tau_med)/len(tau_med), 4),
            "tau_std": round((sum((x-sum(tau_med)/len(tau_med))**2 for x in tau_med)/len(tau_med))**0.5, 4),
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(medians, f, ensure_ascii=False, indent=2)
    print(f"[tau* iop] Saved median τ* to {OUTPUT_PATH}")

    # print summary
    enc = [m for m in medians if "encoder.layers" in m["module_name"]]
    dec_self = [m for m in medians if "decoder.layers" in m["module_name"] and "self_attn" in m["module_name"]]
    dec_cross = [m for m in medians if "decoder.layers" in m["module_name"] and "encoder_attn" in m["module_name"]]
    for label, arr in [("encoder self", enc), ("decoder self", dec_self), ("decoder cross", dec_cross)]:
        taus = [x for m in arr for x in m["tau_per_head"]]
        if taus:
            print(f"  {label}: mean={sum(taus)/len(taus):.2f}, min={min(taus):.2f}, max={max(taus):.2f}")


if __name__ == "__main__":
    main()
