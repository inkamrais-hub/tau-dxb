"""Smoke test: τ-opus fused operator + SparseGrad forward/backward."""
import os, sys, torch

os.environ["HF_HOME"] = "/hy-tmp/hf_cache"
sys.path.insert(0, "/hy-tmp/dimsum/scripts")

from transformers import WhisperForConditionalGeneration
from patch_whisper_stau import apply_learnable_stau
from patch_whisper_sparse_grad import apply_sparse_grad_wta

print("Loading whisper-small (eager) ...")
model = WhisperForConditionalGeneration.from_pretrained(
    "/hy-tmp/whisper-small-local",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

# Apply learnable τ-opus (scalar init for smoke test)
apply_learnable_stau(model, tau_init=1.5, alpha_init=1.0)

# Apply SparseGrad
apply_sparse_grad_wta(model, k_ratio=0.3, target="encoder+decoder")

model = model.cuda()

# Forward pass
feats = torch.randn(1, 80, 3000, dtype=torch.bfloat16).cuda()
ids = torch.randint(0, model.config.vocab_size, (1, 5)).cuda()

print("Forward pass ...")
out = model(input_features=feats, decoder_input_ids=ids)
print(f"  logits shape = {out.logits.shape}")

# Backward pass
print("Backward pass ...")
loss = out.logits.sum()
loss.backward()

# Check τ gradients
tau_grads = []
alpha_grads = []
for name, m in model.named_modules():
    if hasattr(m, "_stau_opus") and m._stau_opus is not None:
        if m._stau_opus.log_tau.grad is not None:
            tau_grads.append(m._stau_opus.log_tau.grad.abs().mean().item())
        if m._stau_opus.log_alpha.grad is not None:
            alpha_grads.append(m._stau_opus.log_alpha.grad.abs().mean().item())

if tau_grads:
    print(f"  tau grad mean: {sum(tau_grads)/len(tau_grads):.6f} ({len(tau_grads)} modules)")
else:
    print("  WARNING: no tau gradients!")
if alpha_grads:
    print(f"  alpha grad mean: {sum(alpha_grads)/len(alpha_grads):.6f} ({len(alpha_grads)} modules)")

print("SMOKE TEST PASSED")
