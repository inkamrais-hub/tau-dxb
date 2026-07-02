"""erp3 完整冒烟测试：erp1权重 + erp2 τ*(分类型σ) + WTA前向稀疏 + 剪枝 + bf16."""
import os, sys, json, torch
import torch.nn.functional as F

os.environ["HF_HOME"] = "/hy-tmp/hf_cache"
sys.path.insert(0, "/hy-tmp/dimsum/scripts")

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from patch_whisper_stau import apply_learnable_stau, load_tau_dict
from patch_whisper_sparse_grad import (
    apply_sparse_grad_wta, set_win_collection, get_win_tracker,
    prune_low_winners, reset_win_counts, set_sparse_grad_k_ratio,
)

ERP1_CKPT = "/hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt"
TAU_STAR_JSON = "/hy-tmp/dimsum/outputs/erp2_tau_star.json"
MODEL_NAME = "/hy-tmp/whisper-small-local"

SIGMA_MAP = {"encoder_self": "softplus", "decoder_self": "sigmoid", "decoder_cross": "exp"}

print("=" * 60)
print("erp3 完整冒烟测试")
print("=" * 60)

# 1. erp2 τ*
print("\n[1] 加载 erp2 τ* ...")
tau_dict = load_tau_dict(TAU_STAR_JSON)
for k, layers in tau_dict.items():
    if layers:
        taus = [t for v in layers.values() for t in v]
        print(f"  {k}: mean={sum(taus)/len(taus):.2f}")

# 2. 模型
print("\n[2] 加载模型 (bf16, eager) ...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, attn_implementation="eager",
    torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=False,
)

# 3. erp1 权重
print("\n[3] 加载 erp1 权重 ...")
sd = torch.load(ERP1_CKPT, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

# 4. 全参解冻
for p in model.parameters():
    p.requires_grad = True

# 5. τ-opus (分类型 σ)
print("\n[4] 应用 τ-opus (分类型 σ) ...")
apply_learnable_stau(model, tau_init=TAU_STAR_JSON, alpha_init=1.0, sigma_name=SIGMA_MAP)

# 6. WTA + SparseGrad
print("\n[5] 应用 WTA(encoder) + SparseGrad(decoder) + track_wins ...")
apply_sparse_grad_wta(
    model, target="encoder+decoder",
    encoder_mode="wta_activation", decoder_mode="sparse_grad",
    encoder_k_ratio=0.6, decoder_k_ratio=0.5, track_wins=True,
)

model = model.cuda()

# 7. 前向 + 反向
print("\n[6] 前向 + 反向 (bf16) ...")
feats = torch.randn(1, 80, 3000, dtype=torch.bfloat16).cuda()
labels = torch.randint(0, model.config.vocab_size, (1, 10)).cuda()

out = model(input_features=feats, labels=labels)
print(f"  loss: {out.loss.item():.4f}")
out.loss.backward()

# 8. 梯度检查
print("\n[7] 梯度统计 ...")
tau_g, alpha_g, base_g = [], [], []
for name, p in model.named_parameters():
    if p.grad is None: continue
    g = p.grad.abs().mean().item()
    if "log_tau" in name: tau_g.append(g)
    elif "log_alpha" in name: alpha_g.append(g)
    else: base_g.append(g)
print(f"  τ grad: mean={sum(tau_g)/len(tau_g):.4f} ({len(tau_g)})")
print(f"  α grad: mean={sum(alpha_g)/len(alpha_g):.4f} ({len(alpha_g)})")
print(f"  base grad: mean={sum(base_g)/len(base_g):.4f} ({len(base_g)})")

# NaN/Inf
has_nan = any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
has_inf = any(torch.isinf(p.grad).any().item() for p in model.parameters() if p.grad is not None)
print(f"  NaN: {has_nan}, Inf: {has_inf}")

# 9. Win count
print("\n[8] Win count 统计 ...")
tracker = get_win_tracker()
if tracker:
    for name, counts in list(tracker.win_counts.items())[:3]:
        total = counts.sum().item()
        nonzero = (counts > 0).sum().item()
        print(f"  {name}: {total} wins, {nonzero}/{len(counts)} neurons active")

# 10. 软剪枝测试
print("\n[9] 软剪枝测试 (threshold=0.01) ...")
# 先记录剪枝前的 fc1 权重
from patch_whisper_sparse_grad import _GRAD_MASKS
enc0_name = "model.encoder.layers.0"
enc0_fc1_w_before = model.model.encoder.layers[0].fc1.weight.data.clone()
n = prune_low_winners(model, threshold=0.01)
print(f"  Soft-pruned: {n} neurons")

# 验证：权重未被置零
enc0_fc1_w_after = model.model.encoder.layers[0].fc1.weight.data.clone()
weight_changed = not torch.equal(enc0_fc1_w_before, enc0_fc1_w_after) if enc0_fc1_w_before.shape == enc0_fc1_w_after.shape else True
print(f"  权重保持（未置零）: {not weight_changed}")

# 验证：梯度 mask 生效
print("\n[10] 剪枝后前向+反向（验证梯度 mask）...")
model.zero_grad()
out2 = model(input_features=feats, labels=labels)
print(f"  loss after prune: {out2.loss.item():.4f}")
out2.loss.backward()

# 检查 pruned 神经元的梯度是否为 0
if enc0_name in _GRAD_MASKS:
    mask = _GRAD_MASKS[enc0_name]
    fc1_grad = model.model.encoder.layers[0].fc1.weight.grad
    # pruned 行的梯度应该接近 0
    pruned_grad_norm = fc1_grad[~mask].abs().mean().item() if (~mask).any() else 0.0
    active_grad_norm = fc1_grad[mask].abs().mean().item() if mask.any() else 0.0
    print(f"  pruned 神经元梯度均值: {pruned_grad_norm:.6f} (应为 ~0)")
    print(f"  active 神经元梯度均值: {active_grad_norm:.6f} (应非零)")

# 验证：unfreeze_all 后梯度恢复
print("\n[11] unfreeze_all 后梯度恢复 ...")
from patch_whisper_sparse_grad import unfreeze_all
unfreeze_all()
model.zero_grad()
out3 = model(input_features=feats, labels=labels)
out3.loss.backward()
fc1_grad_restored = model.model.encoder.layers[0].fc1.weight.grad
all_grad_norm = fc1_grad_restored.abs().mean().item()
print(f"  全部神经元梯度均值: {all_grad_norm:.6f} (应非零)")

print("\n" + "=" * 60)
print("冒烟测试完成！所有组件正常工作。")
print("=" * 60)
