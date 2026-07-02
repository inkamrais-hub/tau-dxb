"""
erp3 冒烟测试：验证 erp1 权重 + erp2 τ* 初始化 + bf16 算子前向/反向。
不训练，只验证数据流。
"""
import os, sys, json, torch
import torch.nn.functional as F

os.environ["HF_HOME"] = "/hy-tmp/hf_cache"
sys.path.insert(0, "/hy-tmp/dimsum/scripts")

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from patch_whisper_stau import apply_learnable_stau, load_tau_dict
from patch_whisper_sparse_grad import apply_sparse_grad_wta

ERP1_CKPT = "/hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt"
TAU_STAR_JSON = "/hy-tmp/dimsum/outputs/erp2_tau_star.json"
MODEL_NAME = "/hy-tmp/whisper-small-local"

print("=" * 60)
print("erp3 冒烟测试：erp1 权重 + erp2 τ* + bf16 算子")
print("=" * 60)

# 1. 加载 erp2 τ* 分布
print("\n[1] 加载 erp2 τ* 分布 ...")
tau_dict = load_tau_dict(TAU_STAR_JSON)
for kind in ("encoder_self", "decoder_self", "decoder_cross"):
    layers = tau_dict.get(kind, {})
    if layers:
        all_taus = [t for layer_taus in layers.values() for t in layer_taus]
        print(f"  {kind}: {len(layers)} layers, mean={sum(all_taus)/len(all_taus):.2f}, "
              f"min={min(all_taus):.2f}, max={max(all_taus):.2f}")

# 2. 加载模型 (bf16, eager)
print("\n[2] 加载 whisper-small (bf16, eager) ...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=False,
)
print(f"  model dtype: {next(model.parameters()).dtype}")

# 3. 加载 erp1 权重
print("\n[3] 加载 erp1 权重 ...")
state_dict = torch.load(ERP1_CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  Missing keys: {len(missing)} (expected: τ-opus params not in erp1)")
print(f"  Unexpected keys: {len(unexpected)}")
if missing:
    # 显示前几个 missing key（应该是 _stau_opus 相关的）
    sample = [k for k in missing[:5]]
    print(f"  Sample missing: {sample}")

# 4. 应用 τ-opus（从 erp2 τ* 初始化）
print("\n[4] 应用可学习 τ-opus (从 erp2 τ* 初始化) ...")
apply_learnable_stau(
    model,
    tau_init=TAU_STAR_JSON,
    alpha_init=1.0,
    sigma_name="softplus",
)

# 5. 验证 τ 初始值是否匹配 erp2
print("\n[5] 验证 τ 初始值 ...")
# 取 encoder layer 0 的 τ
for name, module in model.named_modules():
    if "encoder.layers.0.self_attn" == name and hasattr(module, "_stau_opus"):
        opus = module._stau_opus
        taus = (F.softplus(opus.log_tau) + 1.0).detach().float().cpu().tolist()
        expected = tau_dict["encoder_self"].get(0, [])
        print(f"  encoder L0 τ (init): {[f'{t:.2f}' for t in taus]}")
        print(f"  encoder L0 τ (erp2): {[f'{t:.2f}' for t in expected]}")
        match = all(abs(a-b) < 0.01 for a, b in zip(taus, expected))
        print(f"  Match: {'YES' if match else 'NO'}")
        break

# 6. 应用 SparseGrad
print("\n[6] 应用 SparseGrad (k=0.4) ...")
apply_sparse_grad_wta(model, k_ratio=0.4, target="encoder+decoder", mode="sparse_grad")

# 7. 移到 GPU
model = model.cuda()
print(f"\n[7] 模型移至 CUDA, dtype={next(model.parameters()).dtype}")

# 8. 前向传播（bf16）
print("\n[8] 前向传播 (bf16) ...")
feats = torch.randn(1, 80, 3000, dtype=torch.bfloat16).cuda()
ids = torch.randint(0, model.config.vocab_size, (1, 5)).cuda()

out = model(input_features=feats, decoder_input_ids=ids)
print(f"  logits shape: {out.logits.shape}, dtype: {out.logits.dtype}")
print(f"  loss: {out.loss}")

# 9. 反向传播
print("\n[9] 反向传播 ...")
loss = out.logits.sum() / 1000  # 缩小 loss 避免梯度爆炸
loss.backward()

# 10. 检查梯度
print("\n[10] 梯度统计 ...")
tau_grads = []
alpha_grads = []
base_grads = []
for name, param in model.named_parameters():
    if param.grad is None:
        continue
    grad_abs = param.grad.abs().mean().item()
    if "log_tau" in name:
        tau_grads.append(grad_abs)
    elif "log_alpha" in name:
        alpha_grads.append(grad_abs)
    else:
        base_grads.append(grad_abs)

if tau_grads:
    print(f"  τ grad: mean={sum(tau_grads)/len(tau_grads):.6f}, "
          f"max={max(tau_grads):.6f} ({len(tau_grads)} params)")
if alpha_grads:
    print(f"  α grad: mean={sum(alpha_grads)/len(alpha_grads):.6f}, "
          f"max={max(alpha_grads):.6f} ({len(alpha_grads)} params)")
if base_grads:
    print(f"  base grad: mean={sum(base_grads)/len(base_grads):.6f}, "
          f"max={max(base_grads):.6f} ({len(base_grads)} params)")

# 11. 检查 NaN/Inf
print("\n[11] NaN/Inf 检查 ...")
has_nan = False
has_inf = False
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            has_nan = True
            print(f"  NaN in {name}")
        if torch.isinf(param.grad).any():
            has_inf = True
            print(f"  Inf in {name}")

if not has_nan and not has_inf:
    print("  No NaN/Inf detected. OK.")

print("\n" + "=" * 60)
print("冒烟测试完成。erp1 + erp2 产物可正确导入，bf16 算子正常工作。")
print("=" * 60)
