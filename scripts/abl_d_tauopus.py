"""
Ablation D: τ-opus only (no SparseGrad-WTA).
"""
import os, json
os.environ["HF_HOME"] = "/hf_cache"
os.environ["TORCH_HOME"] = "/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import modeling_whisper as whisper_module
import soundfile as sf
from tqdm import tqdm

TAU_STAR_PATH = "/root/dimsum/outputs/whisper_small_tau_star.json"
OUTPUT_DIR = "/root/dimsum/outputs/abl_d_tauopus"
BATCH_SIZE = 4; GRAD_ACCUM = 4; LR = 1e-5; NUM_EPOCHS = 5
MAX_LENGTH = 128; SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# τ-opus
CLAMP_MIN = 1e-8
def _sf(x): return F.softplus(x).clamp(min=CLAMP_MIN)
def _ss(x): return torch.sigmoid(x).clamp(min=CLAMP_MIN)
def _se(x): return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)
_SIGMA = {"softplus": _sf, "sigmoid": _ss, "exp": _se}
def _spf(x): return torch.sigmoid(x)
def _sps(x): s=torch.sigmoid(x); return s*(1-s)
def _spe(x): return torch.exp(x.clamp(max=15))
_SIGMA_PRIME = {"softplus": _spf, "sigmoid": _sps, "exp": _spe}

class STauOpusFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, tau, sn):
        fn=_SIGMA[sn]; xs=s-s.max(-1,keepdim=True).values
        sv=fn(xs); q=sv.clamp(min=CLAMP_MIN).pow(tau)
        sq=q.sum(-1,keepdim=True).clamp(min=CLAMP_MIN); a=q/sq
        ctx.save_for_backward(s,xs,sv,a,sq); ctx.tau=tau; ctx.sn=sn; return a
    @staticmethod
    def backward(ctx, dO):
        s,xs,sv,a,sq=ctx.saved_tensors; tau,sn=ctx.tau,ctx.sn
        pf=_SIGMA_PRIME.get(sn,lambda x:torch.ones_like(x))
        st=sv.clamp(min=CLAMP_MIN).pow(tau-1); sp=pf(xs)
        A=tau*st*sp; S=sq; q=sv.clamp(min=CLAMP_MIN).pow(tau)
        wda=(dO*q).sum(-1,keepdim=True); da=(dO*A).sum(-1,keepdim=True)
        sA=A.sum(-1,keepdim=True)
        t1=A*(dO/S-wda/S.pow(2))
        t2f=da/S-wda*sA/S.pow(2)
        am=s.argmax(-1,keepdim=True); t2=torch.zeros_like(t1)
        t2.scatter_(-1,am,-t2f)
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

def load_tau_star(path):
    with open(path) as f: data = json.load(f)
    r={"encoder_self":[],"decoder_self":[],"decoder_cross":[]}
    for it in data:
        k="decoder_cross" if it["cross_attention"] else ("decoder_self" if it["is_decoder"] else "encoder_self")
        r[k].append(it["tau_per_head"])
    return r

def apply_tau_opus(model, td, sm):
    et=td.get("encoder_self",[[1.0]*12]*12); dt=td.get("decoder_self",[[1.0]*12]*12)
    ct=td.get("decoder_cross",[[1.0]*12]*12)
    im={"encoder_self":0,"decoder_self":0,"decoder_cross":0}; tm={"encoder_self":et,"decoder_self":dt,"decoder_cross":ct}
    for name,mod in model.named_modules():
        if mod.__class__.__name__!="WhisperAttention": continue
        k="decoder_cross" if ("encoder_attn" in name or "cross" in name.lower()) else ("decoder_self" if getattr(mod,"is_decoder",False) else "encoder_self")
        tl=tm[k][im[k]%len(tm[k])]; im[k]+=1
        tau_t=torch.tensor(tl[:mod.num_heads],dtype=torch.float32).view(1,mod.num_heads,1,1)
        mod.register_buffer("_stau_tau",tau_t); mod._stau_sigma=sm.get(k,"softplus")
    whisper_module.eager_attention_forward=_mk_stau()
    total=sum(im.values())
    print(f"  τ-opus: {total} modules (enc={im['encoder_self']}, dec_self={im['decoder_self']}, cross={im['decoder_cross']})")

class CantoneseASRDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        with open(jsonl_path,"r",encoding="utf-8") as f: self.data=[json.loads(line) for line in f if line.strip()]
        self.processor=processor
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item=self.data[idx]; audio,sr=sf.read(item["audio_path"])
        if sr!=SAMPLING_RATE:
            import librosa; audio=librosa.resample(audio,orig_sr=sr,target_sr=SAMPLING_RATE)
        if audio.ndim>1: audio=audio.mean(-1)
        inp=self.processor.feature_extractor(audio,sampling_rate=SAMPLING_RATE,return_tensors="pt")
        lbl=self.processor.tokenizer(item["text"],truncation=True,max_length=MAX_LENGTH,return_tensors="pt")
        return {"input_features":inp.input_features.squeeze(0),"labels":lbl.input_ids.squeeze(0)}

def collate_fn(batch):
    mf=max(b["input_features"].shape[-1] for b in batch); fd=batch[0]["input_features"].shape[-2]
    inp=torch.zeros(len(batch),fd,mf)
    for i,b in enumerate(batch): inp[i,:,:b["input_features"].shape[-1]]=b["input_features"]
    ml=max(b["labels"].shape[-1] for b in batch)
    lbl=torch.full((len(batch),ml),-100,dtype=torch.long)
    for i,b in enumerate(batch): lbl[i,:b["labels"].shape[-1]]=b["labels"]
    return {"input_features":inp,"labels":lbl}

def main():
    print(f"[Abl D] τ-opus only | Device: {DEVICE}")
    processor=WhisperProcessor.from_pretrained("openai/whisper-small",cache_dir="/hf_cache",language="zh",task="transcribe")
    train_ds=CantoneseASRDataset("/root/dimsum/data/prepared/train.jsonl",processor)
    val_ds=CantoneseASRDataset("/root/dimsum/data/prepared/val.jsonl",processor)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn,num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model=WhisperForConditionalGeneration.from_pretrained("openai/whisper-small",attn_implementation="eager",cache_dir="/hf_cache")
    for p in model.model.encoder.parameters(): p.requires_grad=False
    td=load_tau_star(TAU_STAR_PATH)
    apply_tau_opus(model,td,{"encoder_self":"softplus","decoder_self":"sigmoid","decoder_cross":"exp"})
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")
    model.to(DEVICE)

    optimizer=AdamW(model.model.decoder.parameters(),lr=LR,weight_decay=0.01)
    total_steps=(len(train_loader)*NUM_EPOCHS)//GRAD_ACCUM
    scheduler=CosineAnnealingLR(optimizer,T_max=total_steps,eta_min=1e-6)

    gs,bl=0,float("inf")
    for ep in range(NUM_EPOCHS):
        model.train(); tl=0.0; ns=0; optimizer.zero_grad()
        pbar=tqdm(train_loader,desc=f"Epoch {ep+1}/{NUM_EPOCHS}")
        for bi,batch in enumerate(pbar):
            loss=model(input_features=batch["input_features"].to(DEVICE),labels=batch["labels"].to(DEVICE)).loss/GRAD_ACCUM
            loss.backward()
            if (bi+1)%GRAD_ACCUM==0:
                nn.utils.clip_grad_norm_(model.model.decoder.parameters(),1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); gs+=1
            tl+=loss.item()*GRAD_ACCUM; ns+=1
            if gs%10==0: pbar.set_postfix({"loss":f"{loss.item()*GRAD_ACCUM:.4f}"})
        avg=tl/ns; print(f"\nEpoch {ep+1} avg loss: {avg:.4f}")
        if avg<bl: bl=avg; os.makedirs(os.path.join(OUTPUT_DIR,"best_model"),exist_ok=True); torch.save(model.state_dict(),os.path.join(OUTPUT_DIR,"best_model","model.pt"))
    os.makedirs(os.path.join(OUTPUT_DIR,"final_model"),exist_ok=True); torch.save(model.state_dict(),os.path.join(OUTPUT_DIR,"final_model","model.pt"))
    print("[Abl D] Done!")

if __name__=="__main__": main()
