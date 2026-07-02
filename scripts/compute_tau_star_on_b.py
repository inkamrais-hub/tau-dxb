"""
Compute τ* on B model (pure finetune), 10 runs, aggregate by median.
Fixed: per-module hooks for correct score attribution.
Usage: python scripts/compute_tau_star_on_b.py
"""
import os, json, random
os.environ["HF_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.modeling_whisper import WhisperAttention, eager_attention_forward
from transformers.models.whisper import modeling_whisper as whisper_module
import soundfile as sf
import numpy as np
from tqdm import tqdm

CKPT = "/root/dimsum/outputs/abl_b_pure/best_model/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT = "/root/dimsum/outputs/whisper_small_tau_star_b.json"

# ── Patch eager_attention_forward to capture pre-softmax scores ──
class ScoreCollector:
    """Capture pre-softmax scores by replacing eager_attention_forward.
    Uses module id mapping to distinguish layers."""
    def __init__(self, model):
        self.scores = {}  # module_name -> list of scores
        self.forward_counts = {}  # module_name -> call count
        self._original = None
        # Build name mapping by module id
        self._mod_map = {}
        for name, mod in model.named_modules():
            if isinstance(mod, WhisperAttention):
                self._mod_map[id(mod)] = name

    def _forward(self, module, query, key, value, attention_mask,
                 scaling=None, dropout=0.0, **kwargs):
        if scaling is None:
            scaling = query.size(-1) ** -0.5
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        name = self._mod_map.get(id(module), str(id(module)))
        self.scores.setdefault(name, []).append(scores.detach().float().cpu())
        self.forward_counts[name] = self.forward_counts.get(name, 0) + 1

        # Return original softmax (no modification to model behavior)
        attn_weights = F.softmax(scores.float(), dim=-1).to(value.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=False)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    def install(self):
        self._original = whisper_module.eager_attention_forward
        whisper_module.eager_attention_forward = self._forward

    def uninstall(self):
        if self._original is not None:
            whisper_module.eager_attention_forward = self._original

    def flush(self, name):
        if name not in self.scores or not self.scores[name]:
            return None
        s_list = self.scores[name]
        self.scores[name] = []
        return s_list


# ── τ* solver (ported from τopus/tau_star_opus.py) ──
def solve_tau_star(scores, n_iter=100, lr=0.05):
    """
    scores: (N, T, T) pre-softmax scores for one head
    Find τ minimizing KL(softmax(scores) || σ_softplus(scores)^τ / Z)
    """
    ref = F.softmax(scores, dim=-1).float()

    tau = torch.tensor(2.0, requires_grad=True, device=scores.device)
    opt = torch.optim.SGD([tau], lr=lr)

    best_tau, best_kl = tau.item(), float("inf")
    for _ in range(n_iter):
        opt.zero_grad()
        sigma = F.softplus(scores).clamp(min=1e-8)
        q = sigma.pow(tau)
        Z = q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        p_hat = q / Z
        kl = (ref * (ref / p_hat).clamp(min=1e-8).log()).sum(dim=-1).mean()
        kl.backward()
        opt.step()
        with torch.no_grad():
            tau.clamp_(0.1, 10.0)
        if kl.item() < best_kl:
            best_kl, best_tau = kl.item(), tau.item()
    return best_tau, best_kl


def main():
    print(f"Device: {DEVICE}")

    # Load model
    print("Loading B model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", attn_implementation="eager", cache_dir="/hf_cache",
    )
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", cache_dir="/hf_cache",
        language="zh", task="transcribe",
    )

    # Load estimation pool
    with open("/root/dimsum/data/prepared/val.jsonl") as f:
        all_data = [json.loads(l) for l in f if l.strip()]
    with open("/root/dimsum/data/prepared/train.jsonl") as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    all_data += random.sample(train_data, min(1000, len(train_data)))
    print(f"Estimation pool: {len(all_data)} samples")

    # List all attention modules
    attn_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, WhisperAttention):
            is_dec = getattr(mod, "is_decoder", False)
            is_cross = "encoder_attn" in name or "cross" in name.lower()
            attn_modules.append((name, mod, is_dec, is_cross))
    print(f"Attention modules: {len(attn_modules)}")

    # Install score collector
    collector = ScoreCollector(model)
    collector.install()

    n_runs = 5
    SAMPLES_PER_RUN = 50
    FLUSH_EVERY = 10  # flush and solve every N samples to avoid OOM
    all_tau_lists = []

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...")
        subset = random.sample(all_data, min(SAMPLES_PER_RUN, len(all_data)))

        # Run forward passes to collect scores, flush periodically
        per_run_per_module = {name: [] for name, _, _, _ in attn_modules}  # list of lists (per sample)

        for idx, item in enumerate(tqdm(subset, desc=f"  Run {run+1}")):
            ap = item["audio_path"]
            if not os.path.exists(ap): continue
            try:
                audio, sr = sf.read(ap)
                if sr != 16000:
                    import librosa; audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                if audio.ndim > 1: audio = audio.mean(-1)
            except: continue

            inputs = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(DEVICE)

            with torch.no_grad():
                _ = model.generate(inputs, language="zh", task="transcribe",
                                   num_beams=1, temperature=0.0)

            # Flush scores per module and store as list of per-sample tensors
            for name, _, _, _ in attn_modules:
                scores_list = collector.flush(name)
                if scores_list:
                    per_run_per_module[name].extend(scores_list)  # list of (1, nH, T_i, T_i)

        # Solve τ* per head: iterate per-sample, aggregate by median
        run_tau_list = []
        for name, mod, _, _ in attn_modules:
            scores_samples = per_run_per_module.get(name, [])
            if len(scores_samples) < 3:
                run_tau_list.append([1.0] * mod.num_heads)
                continue

            nH = mod.num_heads
            tau_heads = []
            for h in range(nH):
                tau_per_sample = []
                for s in scores_samples:  # s: (1, nH, T_i, T_i)
                    hs = s[0, h:h+1, :, :]  # (1, T_i, T_i), keep batch dim
                    if hs.shape[-1] < 4:  # too short
                        continue
                    tau, kl = solve_tau_star(hs)
                    tau_per_sample.append(tau)
                if len(tau_per_sample) < 3:
                    tau_heads.append(1.0)
                else:
                    tau_heads.append(round(float(np.median(tau_per_sample)), 4))
            run_tau_list.append(tau_heads)
            if run == 0:
                print(f"  {name.split('.')[-1]}: τ={[f'{t:.2f}' for t in tau_heads[:4]]}...")

        all_tau_lists.append(run_tau_list)

    collector.uninstall()

    # Aggregate by median
    print(f"\n{'='*50}")
    print(f"Aggregating τ* across {n_runs} runs (median) ...")

    results = []
    for idx, (name, mod, is_dec, is_cross) in enumerate(attn_modules):
        nH = mod.num_heads
        tau_matrix = []
        for run_idx in range(n_runs):
            if idx < len(all_tau_lists[run_idx]):
                tau_matrix.append(all_tau_lists[run_idx][idx][:nH])

        if tau_matrix:
            ta = np.array(tau_matrix)
            tau_med = np.median(ta, axis=0).tolist()
            tau_mn = np.mean(ta, axis=0).tolist()
            tau_sd = np.std(ta, axis=0).tolist()
        else:
            tau_med = [1.0] * nH; tau_mn = [1.0] * nH; tau_sd = [0.0] * nH

        results.append({
            "module_name": name, "is_decoder": is_dec,
            "cross_attention": is_cross, "num_heads": nH,
            "tau_per_head": [round(t, 4) for t in tau_med],
            "tau_mean": [round(t, 4) for t in tau_mn],
            "tau_std": [round(t, 4) for t in tau_sd],
        })
        print(f"  {name.split('.')[-1]:30s} τ*_med={np.mean(tau_med):.2f}")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT}")

    enc = [r["tau_per_head"] for r in results if not r["is_decoder"]]
    dec = [r["tau_per_head"] for r in results if r["is_decoder"] and not r["cross_attention"]]
    crs = [r["tau_per_head"] for r in results if r["cross_attention"]]
    print(f"\nSummary:")
    print(f"  Encoder self:   mean={np.mean(enc):.2f} ± {np.std(enc):.2f}" if enc else "  Encoder self:  N/A")
    print(f"  Decoder self:   mean={np.mean(dec):.2f} ± {np.std(dec):.2f}" if dec else "  Decoder self:  N/A")
    print(f"  Decoder cross:  mean={np.mean(crs):.2f} ± {np.std(crs):.2f}" if crs else "  Decoder cross: N/A")

if __name__ == "__main__":
    main()
