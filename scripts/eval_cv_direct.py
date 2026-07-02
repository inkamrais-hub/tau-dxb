"""Download CV test audio files directly from HF Hub, then evaluate Base-XL."""
import os, json
os.environ["HF_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module
from huggingface_hub import hf_hub_download, list_repo_files
import soundfile as sf
from tqdm import tqdm

CKPT = "/root/dimsum/outputs/integrated/best_model/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "/root/dimsum/outputs/cv_test_results.json"

# τ-opus (inlined)
CLAMP_MIN = 1e-8
def _sf(x): return F.softplus(x).clamp(min=CLAMP_MIN)
def _ss(x): return torch.sigmoid(x).clamp(min=CLAMP_MIN)
def _se(x): return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)
_SIGMA = {"softplus": _sf, "sigmoid": _ss, "exp": _se}
def _spf(x): return torch.sigmoid(x)
def _sps(x): s = torch.sigmoid(x); return s * (1 - s)
def _spe(x): return torch.exp(x.clamp(max=15))
_SIGMA_PRIME = {"softplus": _spf, "sigmoid": _sps, "exp": _spe}

class STauOpusFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, tau, sn):
        fn = _SIGMA[sn]; xs = s - s.max(-1,keepdim=True).values
        sv = fn(xs); q = sv.clamp(min=CLAMP_MIN).pow(tau)
        sq = q.sum(-1,keepdim=True).clamp(min=CLAMP_MIN); a = q/sq
        ctx.save_for_backward(s, xs, sv, a, sq); ctx.tau = tau; ctx.sn = sn; return a
    @staticmethod
    def backward(ctx, dO):
        s, xs, sv, a, sq = ctx.saved_tensors; tau, sn = ctx.tau, ctx.sn
        pf = _SIGMA_PRIME.get(sn, lambda x: torch.ones_like(x))
        st = sv.clamp(min=CLAMP_MIN).pow(tau-1); sp = pf(xs)
        A = tau * st * sp; S = sq; q = sv.clamp(min=CLAMP_MIN).pow(tau)
        wda = (dO*q).sum(-1,keepdim=True); da = (dO*A).sum(-1,keepdim=True)
        sA = A.sum(-1,keepdim=True)
        t1 = A*(dO/S - wda/S.pow(2))
        t2f = da/S - wda*sA/S.pow(2)
        am = s.argmax(-1,keepdim=True); t2 = torch.zeros_like(t1)
        t2.scatter_(-1, am, -t2f)
        return (t1+t2).float(), None, None

def _mk_stau():
    def f(m, q, k, v, mask, scaling=None, dropout=0.0, **kw):
        if scaling is None: scaling = q.size(-1)**-0.5
        s = torch.matmul(q, k.transpose(2,3))*scaling
        if mask is not None: s = s+mask
        t = getattr(m, "_stau_tau", torch.tensor(1.0, device=s.device)).to(s.device)
        sn = getattr(m, "_stau_sigma", "softplus")
        aw = STauOpusFn.apply(s.float(), t, sn)
        aw = F.dropout(aw, p=dropout, training=m.training)
        o = torch.matmul(aw.to(v.dtype), v).transpose(1,2).contiguous()
        return o, aw.to(v.dtype)
    return f

def main():
    print(f"Device: {DEVICE}")

    # List repo files to find test audio
    print("Listing CV repo files...")
    all_files = list_repo_files("kaschung4/common_voice_17_yue_pseudo_labelled", repo_type="dataset")
    test_parquets = [f for f in all_files if "test" in f and f.endswith(".parquet")]
    print(f"Test parquets: {len(test_parquets)}")

    # Download and read test parquet
    import pyarrow.parquet as pq
    texts = []
    audio_rel_paths = []
    for pq_file in sorted(test_parquets)[:5]:  # max 5 shards
        local = hf_hub_download("kaschung4/common_voice_17_yue_pseudo_labelled",
                                pq_file, repo_type="dataset",
                                local_dir="/hf_cache/cv_yue", local_dir_use_symlinks=False)
        table = pq.read_table(local, columns=["audio", "sentence"])
        audio_paths = table.column("audio").to_pylist()
        sent_col = table.column("sentence").to_pylist()
        for j in range(len(table)):
            audio_info = json.loads(audio_paths[j]) if isinstance(audio_paths[j], str) else audio_paths[j]
            audio_rel_paths.append(audio_info.get("path", ""))
            texts.append(sent_col[j])

    print(f"Samples: {len(texts)}")
    if len(texts) == 0:
        print("ERROR: No test samples found. Check parquet structure.")
        return

    # Download actual audio files
    audio_dir = "/hf_cache/cv_yue/audio"
    os.makedirs(audio_dir, exist_ok=True)
    local_paths = []
    for rel_path in tqdm(audio_rel_paths, desc="Downloading audio"):
        local = hf_hub_download("kaschung4/common_voice_17_yue_pseudo_labelled",
                                rel_path, repo_type="dataset",
                                local_dir="/hf_cache/cv_yue",
                                local_dir_use_symlinks=False)
        local_paths.append(local)

    # Load model
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", cache_dir="/hf_cache",
                                                  language="zh", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small",
                                                             attn_implementation="eager", cache_dir="/hf_cache")
    sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd, strict=False)
    sigma_map = {"encoder_self": "softplus", "decoder_self": "sigmoid", "decoder_cross": "exp"}
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "WhisperAttention": continue
        tk = name + "._stau_tau"
        if tk not in sd: continue
        module.register_buffer("_stau_tau", sd[tk].to(DEVICE))
        k = "decoder_cross" if "encoder_attn" in name else ("decoder_self" if getattr(module, "is_decoder", False) else "encoder_self")
        module._stau_sigma = sigma_map[k]; count += 1
    whisper_module.eager_attention_forward = _mk_stau()
    print(f"τ-opus: {count} modules")
    model.to(DEVICE).eval()

    gen_kwargs = {"language": "zh", "task": "transcribe", "num_beams": 5, "temperature": 0.0, "do_sample": False}

    # Inference
    refs, hyps = [], []
    for i, fpath in enumerate(tqdm(local_paths, desc="CV Eval")):
        audio, sr = sf.read(fpath)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.ndim > 1: audio = audio.mean(-1)
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        with torch.no_grad():
            pred_ids = model.generate(inputs, **gen_kwargs)
            text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        refs.append(texts[i]); hyps.append(text)
        if (i+1) % 100 == 0:
            import Levenshtein
            ed = sum(Levenshtein.distance(r, h) for r, h in zip(refs, hyps))
            tc = sum(len(r) for r in refs)
            print(f"  [{i+1}/{len(texts)}] Interim CER: {ed/tc:.4f}")

    # Final
    import Levenshtein
    ed = sum(Levenshtein.distance(r, h) for r, h in zip(refs, hyps))
    tc = sum(len(r) for r in refs)
    cer = ed / tc if tc > 0 else 1.0
    sa = sum(1 for r, h in zip(refs, hyps) if r == h) / len(refs)
    results = {"dataset": "Common Voice 17 yue (test)", "num": len(refs), "cer": cer, "sent_acc": sa,
               "correct": sum(1 for r, h in zip(refs, hyps) if r == h), "total": len(refs)}
    print(f"\n{'='*50}\nCER: {cer:.4f} ({cer*100:.2f}%)\nSent Acc: {sa:.4f}\nCorrect: {results['correct']}/{results['total']}")
    with open(OUTPUT, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
