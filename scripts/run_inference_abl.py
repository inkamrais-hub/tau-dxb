#!/usr/bin/env python3
"""
Unified inference for all ablation models.
Usage: python scripts/run_inference_abl.py <ckpt_dir> <variant> [output_name]
  variant: B | C | D | E
"""
import os, sys, json, subprocess
os.environ["HF_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ckpt_dir = sys.argv[1]
variant = sys.argv[2].upper()
output_name = sys.argv[3] if len(sys.argv) > 3 else f"abl_{variant}"

OUTPUT_DIR = f"/root/dimsum/outputs/{output_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module
import soundfile as sf
from tqdm import tqdm

TEST_JSONL = "/root/dimsum/data/test.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── τ-opus (for D, E) ──
CLAMP_MIN = 1e-8
_SIGMA = {
    "softplus": lambda x: F.softplus(x).clamp(min=CLAMP_MIN),
    "sigmoid": lambda x: torch.sigmoid(x).clamp(min=CLAMP_MIN),
    "exp": lambda x: torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN),
}
_SIGMA_PRIME = {
    "softplus": torch.sigmoid,
    "sigmoid": lambda s: s * (1 - s),
    "exp": lambda x: torch.exp(x.clamp(max=15)),
}

class STauOpusFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, tau, sn):
        fn=_SIGMA[sn]; xs=s-s.max(-1,keepdim=True).values; sv=fn(xs)
        q=sv.clamp(min=CLAMP_MIN).pow(tau); sq=q.sum(-1,keepdim=True).clamp(min=CLAMP_MIN)
        ctx.save_for_backward(s,xs,sv,q/sq,sq); ctx.tau=tau; ctx.sn=sn; return q/sq
    @staticmethod
    def backward(ctx, dO):
        s,xs,sv,a,sq=ctx.saved_tensors; tau,sn=ctx.tau,ctx.sn
        pf=_SIGMA_PRIME.get(sn,lambda x:torch.ones_like(x))
        st=sv.clamp(min=CLAMP_MIN).pow(tau-1); sp=pf(xs); A=tau*st*sp
        S=sq; q=a*sq; wda=(dO*q).sum(-1,keepdim=True); da=(dO*A).sum(-1,keepdim=True)
        sA=A.sum(-1,keepdim=True)
        t1=A*(dO/S-wda/S.pow(2)); t2f=da/S-wda*sA/S.pow(2)
        am=s.argmax(-1,keepdim=True); t2=torch.zeros_like(t1); t2.scatter_(-1,am,-t2f)
        return (t1+t2).float(),None,None

def _mk_stau():
    def f(m,q,k,v,mask,scaling=None,dropout=0.0,**kw):
        if scaling is None: scaling=q.size(-1)**-0.5
        s=torch.matmul(q,k.transpose(2,3))*scaling
        if mask is not None: s=s+mask
        t=getattr(m,"_stau_tau",torch.tensor(1.0,device=s.device)).to(s.device)
        sn=getattr(m,"_stau_sigma","softplus")
        aw=STauOpusFn.apply(s.float(),t,sn)
        aw=F.dropout(aw,p=dropout,training=m.training)
        o=torch.matmul(aw.to(v.dtype),v).transpose(1,2).contiguous()
        return o,aw.to(v.dtype)
    return f

def apply_tau_opus(model):
    with open("/root/dimsum/outputs/whisper_small_tau_star.json") as f: td=json.load(f)
    sigma_map={"encoder_self":"softplus","decoder_self":"sigmoid","decoder_cross":"exp"}
    im={"encoder_self":0,"decoder_self":0,"decoder_cross":0}; tm={"encoder_self":[[1.0]*12]*12,"decoder_self":[[1.0]*12]*12,"decoder_cross":[[1.0]*12]*12}
    for it in td:
        k="decoder_cross" if it["cross_attention"] else ("decoder_self" if it["is_decoder"] else "encoder_self")
        tm[k][im[k]]=it["tau_per_head"]; im[k]+=1
    im2={"encoder_self":0,"decoder_self":0,"decoder_cross":0}
    for name,mod in model.named_modules():
        if mod.__class__.__name__!="WhisperAttention": continue
        k="decoder_cross" if ("encoder_attn" in name or "cross" in name.lower()) else ("decoder_self" if getattr(mod,"is_decoder",False) else "encoder_self")
        tl=tm[k][im2[k]%len(tm[k])]; im2[k]+=1
        tau_t=torch.tensor(tl[:mod.num_heads],dtype=torch.float32).view(1,mod.num_heads,1,1)
        mod.register_buffer("_stau_tau",tau_t); mod._stau_sigma=sigma_map.get(k,"softplus")
    whisper_module.eager_attention_forward=_mk_stau()
    total=sum(im2.values())
    print(f"  τ-opus: {total} modules")

# ── SparseGrad (for C, E) ──
class SGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,k): _,ti=x.topk(k,dim=-1); ctx.save_for_backward(ti); ctx.xs=x.shape; return x
    @staticmethod
    def backward(ctx,gO): ti,=ctx.saved_tensors; m=torch.zeros(ctx.xs,device=gO.device,dtype=gO.dtype); m.scatter_(-1,ti,1.0); return gO*m,None

class SGradMod(nn.Module):
    def __init__(self,fn,k=0.3): super().__init__(); self.fn=fn; self.k=k
    def forward(self,x): x=self.fn(x); return SGradFn.apply(x,max(1,int(x.shape[-1]*self.k))) if self.training else x

def apply_sparse(model,k=0.3):
    from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer, WhisperDecoderLayer
    cnt=0
    for m in model.modules():
        if isinstance(m,(WhisperEncoderLayer,WhisperDecoderLayer)): m.activation_fn=SGradMod(m.activation_fn,k); cnt+=1
    print(f"  SparseGrad: {cnt} layers")

# ── Main ──
def main():
    print(f"Inference [{variant}] ckpt={ckpt_dir} | Device={DEVICE}")

    # Setup model
    if variant in ("D", "E"):
        attn = "eager"
    else:
        attn = "sdpa"

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", attn_implementation=attn, cache_dir="/hf_cache",
    )

    if variant in ("C", "E"):
        apply_sparse(model, 0.3)
    if variant in ("D", "E"):
        apply_tau_opus(model)

    # Load weights
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"  Model loaded from {ckpt_path}")

    # Load test set
    print("Loading test set...")
    with open(TEST_JSONL) as f:
        test_rows = [json.loads(l) for l in f if l.strip()]
    print(f"  Test samples: {len(test_rows)}")

    # Find all audio files recursively (handles both short names and original names)
    import subprocess as sp
    result = sp.run(
        ["find", "/root/dimsum/data/audio_data", "-name", "*.wav"],
        capture_output=True, text=True, timeout=60
    )
    wav_map = {}
    for wav in result.stdout.strip().split("\n"):
        if not wav: continue
        fname = os.path.basename(wav)
        if fname not in wav_map:
            wav_map[fname] = wav
    print(f"  Audio files found: {len(wav_map)}")

    # Use val set for CER evaluation (has ground truth text)
    # Also produce test.jsonl submission for competition
    VAL_JSONL = "/root/dimsum/data/prepared/val.jsonl"
    print(f"  Val samples for eval: {len(open(VAL_JSONL).readlines())}")

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", cache_dir="/hf_cache", language="zh", task="transcribe",
    )

    gen_kwargs = {
        "language": "zh", "task": "transcribe",
        "num_beams": 5, "temperature": 0.0, "do_sample": False,
    }

    # ── Step 1: Inference on val set for CER ──
    with open(VAL_JSONL) as f:
        val_rows = [json.loads(l) for l in f if l.strip()]

    val_results = []
    for item in tqdm(val_rows, desc=f"Val Inference [{variant}]"):
        audio_path = item["audio_path"]
        if not os.path.exists(audio_path):
            val_results.append({"utt_id": item.get("utt_id",""), "pred_text": "", "ref_text": item["text"], "error": "audio not found"})
            continue
        try:
            audio, sr = sf.read(audio_path)
            if sr != 16000: import librosa; audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if audio.ndim > 1: audio = audio.mean(-1)
        except Exception as e:
            val_results.append({"utt_id": item.get("utt_id",""), "pred_text": "", "ref_text": item["text"], "error": str(e)})
            continue
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        with torch.no_grad():
            pred_ids = model.generate(inputs, **gen_kwargs)
            text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        val_results.append({"utt_id": item.get("utt_id",""), "pred_text": text, "ref_text": item["text"]})

    # Save val predictions for evaluator
    val_pred_path = os.path.join(OUTPUT_DIR, "val_pred.jsonl")
    with open(val_pred_path, "w", encoding="utf-8") as f:
        for r in val_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute CER manually (Levenshtein)
    import Levenshtein
    refs = [r["ref_text"] for r in val_results if not r.get("error")]
    preds = [r["pred_text"] for r in val_results if not r.get("error")]
    if refs:
        ed = sum(Levenshtein.distance(r, h) for r, h in zip(refs, preds))
        tc = sum(len(r) for r in refs)
        val_cer = ed / tc if tc else 1.0
        val_sa = sum(1 for r, h in zip(refs, preds) if r == h) / len(refs)
        print(f"\n[Val Set] CER: {val_cer:.4f} ({val_cer*100:.2f}%) | Sent Acc: {val_sa:.4f}")

        val_metrics = {"variant": variant, "num_samples": len(refs), "cer": val_cer, "sentence_accuracy": val_sa}
        with open(os.path.join(OUTPUT_DIR, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f, indent=2)

    # ── Step 2: Inference on test set for submission ──
    print(f"\nTest set inference ({len(test_rows)} samples)...")
    test_results = []
    for item in tqdm(test_rows, desc=f"Test Inference [{variant}]"):
        audio_path = item.get("audio_path", "")
        fname = os.path.basename(audio_path) if audio_path else ""
        ap = wav_map.get(fname, "") if fname else ""
        if not ap or not os.path.exists(ap):
            test_results.append({"key": item.get("utt_id",""), "predict": "", "ground_truth": ""})
            continue
        try:
            audio, sr = sf.read(ap)
            if sr != 16000: import librosa; audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if audio.ndim > 1: audio = audio.mean(-1)
        except:
            test_results.append({"key": item.get("utt_id",""), "predict": "", "ground_truth": ""})
            continue
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        with torch.no_grad():
            pred_ids = model.generate(inputs, **gen_kwargs)
            text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        test_results.append({"key": item.get("utt_id",""), "predict": text, "ground_truth": ""})

    sub_path = os.path.join(OUTPUT_DIR, "submission.jsonl")
    with open(sub_path, "w", encoding="utf-8") as f:
        for r in test_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {sub_path} ({len(test_results)} predictions)")

    # ── Step 3: Run evaluator on test set ──
    eval_dir = os.path.join(OUTPUT_DIR, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    cmd = (f"cd /root/dimsum && python3.11 data/evaluator.py --pred_jsonl {sub_path} "
           f"--report_dir {eval_dir} --prediction_field predict --reference_field ground_truth")
    try:
        ret = sp.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print(ret.stdout[-500:] if len(ret.stdout) > 500 else ret.stdout)
    except Exception as e:
        print(f"  Evaluator error: {e}")

    print(f"[{variant}] Done!")

if __name__ == "__main__":
    import re
    main()
