import os
os.environ['HF_HOME'] = r'F:\hf_cache'
os.environ['TORCH_HOME'] = r'F:\hf_cache'

import torch
from transformers import WhisperForConditionalGeneration
from patch_whisper_stau import apply_learnable_stau, apply_fixed_stau, remove_stau, load_tau_dict
from patch_whisper_sparse_grad import apply_sparse_grad_wta, set_sparse_grad_k_ratio, remove_sparse_grad

print('Loading model...')
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small', attn_implementation='eager', cache_dir=r'F:\hf_cache')
print('Applying learnable stau...')
apply_learnable_stau(model, tau_init=1.0)
print('Applying sparse grad...')
apply_sparse_grad_wta(model, k_ratio=0.5)
print('Set k_ratio 0.3...')
set_sparse_grad_k_ratio(model, 0.3)
print('Forward...')
dummy_features = torch.randn(1, 80, 150)
dummy_ids = torch.randint(0, model.config.vocab_size, (1, 10))
with torch.no_grad():
    out = model(input_features=dummy_features, decoder_input_ids=dummy_ids)
print('OK logits:', out.logits.shape)
remove_sparse_grad(model)
remove_stau(model)
print('Removed patches.')

print('Test load_tau_dict...')
d = load_tau_dict(r'F:\τ\点心杯\outputs\base_iop_tau_star.json')
for k in d:
    print(k, len(d[k]), 'layers')
