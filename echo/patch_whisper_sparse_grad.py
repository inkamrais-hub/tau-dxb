"""
SparseGrad-WTA + 竞争稀疏激活 + 软剪枝 — 接入 Whisper-small FFN。
兼容 transformers >= 5.12。

三种模式（可按 encoder/decoder 分别配置）：
  - "sparse_grad":     前向恒等，反向只传 top-k 梯度（decoder 用）
  - "wta_activation":  前向 top-k 稀疏 + 反向 top-k 梯度（encoder 用）
  - "none":            不做稀疏（跳过）

软剪枝（动态可复活）：
  - 每步记录 win count（WTA 选中的神经元）
  - prune_low_winners() 只冻结低 win rate 神经元的梯度（权重保持原值）
  - 前向不封死：pruned 神经元仍参与 WTA 选择，若被选中可自然复活
  - unfreeze_all() 清空梯度 mask，全部恢复训练
"""
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from collections import defaultdict

# Module-level Triton import — NOT inside hot path
try:
    from triton_wta import wta_fused_forward as _triton_wta_forward
    _HAS_TRITON_WTA = True
except ImportError:
    _triton_wta_forward = None
    _HAS_TRITON_WTA = False


# ═══════════════════════════════════════════════════════════════
# Autograd Functions
# ═══════════════════════════════════════════════════════════════

class SparseGradWTA_Function(torch.autograd.Function):
    """前向恒等，反向只传 top-k 梯度。
    
    优化：用 kthvalue + 比较替代 topk + scatter，避免排序开销。
    kthvalue 是 O(n) 期望时间，topk 是 O(n·log(k))。
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int):
        if k < 1:
            k = max(1, int(x.shape[-1] * 0.3))
        D = x.shape[-1]
        k = min(k, D)
        # kthvalue 找第 (D-k+1) 小的值 = 第 k 大的值（阈值）
        # 比 topk 快因为不需要排序，只需要 partition
        threshold = x.kthvalue(D - k + 1, dim=-1, keepdim=True).values
        mask = (x >= threshold)  # bool mask, no scatter needed
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype), None


class WTA_Activation_Function(torch.autograd.Function):
    """前向 top-k 稀疏 + 反向 top-k 梯度（同一 mask）。"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int):
        if k < 1:
            k = max(1, int(x.shape[-1] * 0.3))
        k = min(k, x.shape[-1])
        _, topk_idx = x.topk(k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)
        ctx.save_for_backward(topk_idx)
        ctx.x_shape = x.shape
        return x * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        topk_idx, = ctx.saved_tensors
        mask = torch.zeros(ctx.x_shape, device=grad_output.device, dtype=grad_output.dtype)
        mask.scatter_(-1, topk_idx, 1.0)
        return grad_output * mask, None


# ═══════════════════════════════════════════════════════════════
# Win count tracker（用于剪枝统计）
# ═══════════════════════════════════════════════════════════════

class WinCountTracker:
    """跟踪每个 FFN 神经元的 win count。"""
    def __init__(self):
        self.win_counts: Dict[str, torch.Tensor] = {}  # module_name -> (d_ff,) tensor
        self.total_steps: Dict[str, int] = defaultdict(int)

    def record(self, module_name: str, topk_idx: torch.Tensor, d_ff: int):
        """记录一次 WTA 选择的赢家。
        topk_idx: (B, Lq, k) — 每个位置选的 top-k 索引
        """
        if module_name not in self.win_counts:
            self.win_counts[module_name] = torch.zeros(d_ff, dtype=torch.long)
        # 统计每个神经元被选中的次数
        flat_idx = topk_idx.reshape(-1).cpu()
        self.win_counts[module_name].scatter_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=torch.long)
        )
        self.total_steps[module_name] += 1

    def win_rate(self, module_name: str) -> Optional[torch.Tensor]:
        """win_rate = win_count / mean(win_count)。
        均匀分布时 win_rate=1.0，高频 >>1，低频 <<1。
        threshold=0.05 意味着 win_count < 0.05 * mean 的被剪。
        """
        if module_name not in self.win_counts:
            return None
        counts = self.win_counts[module_name].float()
        mean_count = counts.mean().clamp(min=1.0)
        return counts / mean_count

    def reset(self):
        self.win_counts.clear()
        self.total_steps.clear()


# ═══════════════════════════════════════════════════════════════
# Patch 系统：支持 encoder/decoder 不同模式
# ═══════════════════════════════════════════════════════════════

_HOOK_HANDLES = []
_K_RATIOS: Dict[str, float] = {"encoder": 0.5, "decoder": 0.5}
_MODES: Dict[str, str] = {"encoder": "sparse_grad", "decoder": "sparse_grad"}
_TRACKER: WinCountTracker = None
_GRAD_MASKS: Dict[str, torch.Tensor] = {}  # module_name -> bool mask (True = 活跃，False = 冻结梯度)
_COLLECT_WINS = False

# ─── 可学习 k_ratio ──────────────────────────────────────────
_LEARNABLE_K: Dict[str, "LearnableKRatio"] = {}  # module_name -> LearnableKRatio


class LearnableKRatio(nn.Module):
    """Per-layer learnable k_ratio for SparseGrad/WTA.

    k_ratio = sigmoid(logit_k), bounded in [0, 1).
    Gradient flows through the sparsity regularization term in the training loss.
    """
    def __init__(self, init_k_ratio: float = 0.4):
        super().__init__()
        self.logit_k = nn.Parameter(torch.tensor(0.0))
        self._reset(init_k_ratio)

    def _reset(self, init_k_ratio: float):
        init_k_ratio = max(0.1, min(0.9, init_k_ratio))
        init_val = math.log(init_k_ratio / (1.0 - init_k_ratio + 1e-8))
        nn.init.constant_(self.logit_k, float(init_val))

    @property
    def k_ratio(self) -> float:
        return torch.sigmoid(self.logit_k).item()

    def get_k(self, d_ff: int) -> int:
        k = int(round(torch.sigmoid(self.logit_k).item() * d_ff))
        return max(1, min(d_ff, k))

    def extra_repr(self) -> str:
        return f"k_ratio={self.k_ratio:.3f}"


def _make_fc1_weight_grad_hook(module_name: str):
    """fc1.weight: (d_ff, d_model)，按行剪。"""
    def hook(grad):
        if module_name in _GRAD_MASKS:
            mask = _GRAD_MASKS[module_name].to(grad.device, grad.dtype)
            return grad * mask.unsqueeze(-1)
        return grad
    return hook


def _make_fc1_bias_grad_hook(module_name: str):
    """fc1.bias: (d_ff,)，按元素剪。"""
    def hook(grad):
        if module_name in _GRAD_MASKS:
            mask = _GRAD_MASKS[module_name].to(grad.device, grad.dtype)
            return grad * mask
        return grad
    return hook


def _make_fc2_weight_grad_hook(module_name: str):
    """fc2.weight: (d_model, d_ff)，按列剪。"""
    def hook(grad):
        if module_name in _GRAD_MASKS:
            mask = _GRAD_MASKS[module_name].to(grad.device, grad.dtype)
            return grad * mask.unsqueeze(0)
        return grad
    return hook


def _make_fc2_pre_hook(module_name: str, layer_type: str):
    """为 fc2 创建 pre-forward hook。前向不应用剪枝 mask（软剪枝：权重保持，前向通过 WTA 自然选择）。"""
    def hook(module, input_):
        x = input_[0]
        if not x.requires_grad:
            return input_

        d_ff = x.shape[-1]
        mode = _MODES.get(layer_type, "sparse_grad")

        # 可学习 k_ratio 优先
        if module_name in _LEARNABLE_K:
            k = _LEARNABLE_K[module_name].get_k(d_ff)
        else:
            k_ratio = _K_RATIOS.get(layer_type, 0.3)
            k = max(1, int(d_ff * k_ratio))

        if mode == "sparse_grad":
            x = SparseGradWTA_Function.apply(x, k)
        elif mode == "wta_activation":
            # 记录 win count（只在收集模式）
            if _COLLECT_WINS and _TRACKER is not None:
                with torch.no_grad():
                    _, topk_idx = x.topk(k, dim=-1)
                    _TRACKER.record(module_name, topk_idx, d_ff)
            # Triton fused kernel (auto fallback to PyTorch if not available)
            if _HAS_TRITON_WTA:
                x = _triton_wta_forward(x, k)
            else:
                x = WTA_Activation_Function.apply(x, k)
        # mode == "none": 不做任何操作

        return (x,) + input_[1:]
    return hook


def apply_sparse_grad_wta(
    model,
    k_ratio: float = 0.3,
    target: str = "encoder+decoder",
    mode: str = "sparse_grad",
    encoder_mode: Optional[str] = None,
    decoder_mode: Optional[str] = None,
    encoder_k_ratio: Optional[float] = None,
    decoder_k_ratio: Optional[float] = None,
    track_wins: bool = False,
    learnable_k: bool = False,
    init_k_ratio: float = 0.4,
):
    """给 whisper 的 FFN 层注入稀疏梯度/激活。

    Args:
        model: WhisperForConditionalGeneration
        k_ratio: 默认 k 比例
        target: "encoder" | "decoder" | "encoder+decoder"
        mode: 默认模式 ("sparse_grad" / "wta_activation" / "none")
        encoder_mode: encoder 专用模式（覆盖 mode）
        decoder_mode: decoder 专用模式（覆盖 mode）
        encoder_k_ratio: encoder 专用 k 比例
        decoder_k_ratio: decoder 专用 k 比例
        track_wins: 是否跟踪 win count（用于剪枝）
        learnable_k: 使用可学习 k_ratio（每层独立）
        init_k_ratio: 可学习 k 的初始值
    """
    global _K_RATIOS, _MODES, _TRACKER, _COLLECT_WINS, _LEARNABLE_K

    # 设置模式和 k_ratio
    enc_mode = encoder_mode or mode
    dec_mode = decoder_mode or mode
    enc_k = encoder_k_ratio or k_ratio
    dec_k = decoder_k_ratio or k_ratio

    _K_RATIOS = {"encoder": enc_k, "decoder": dec_k}
    _MODES = {"encoder": enc_mode, "decoder": dec_mode}

    if track_wins:
        _TRACKER = WinCountTracker()
        _COLLECT_WINS = True
    else:
        _COLLECT_WINS = False

    remove_sparse_grad()
    _LEARNABLE_K = {}

    target_encoder = target in ("encoder", "encoder+decoder")
    target_decoder = target in ("decoder", "encoder+decoder")

    n_enc, n_dec = 0, 0
    for name, module in model.named_modules():
        clsname = module.__class__.__name__
        if clsname == "WhisperEncoderLayer" and target_encoder:
            if hasattr(module, "fc2"):
                # 可学习 k_ratio
                if learnable_k:
                    lk = LearnableKRatio(init_k_ratio=init_k_ratio)
                    module.register_module(f"_learnable_k", lk)
                    _LEARNABLE_K[name] = lk
                h = module.fc2.register_forward_pre_hook(
                    _make_fc2_pre_hook(name, "encoder"), with_kwargs=False
                )
                _HOOK_HANDLES.append(h)
                # 注册梯度 hook（只注册一次，hook 内部从全局 _GRAD_MASKS 读取）
                if not getattr(module, "_grad_hook_installed", False):
                    module.fc1.weight.register_hook(_make_fc1_weight_grad_hook(name))
                    if module.fc1.bias is not None:
                        module.fc1.bias.register_hook(_make_fc1_bias_grad_hook(name))
                    module.fc2.weight.register_hook(_make_fc2_weight_grad_hook(name))
                    module._grad_hook_installed = True
                n_enc += 1
        elif clsname == "WhisperDecoderLayer" and target_decoder:
            if hasattr(module, "fc2"):
                # 可学习 k_ratio
                if learnable_k:
                    lk = LearnableKRatio(init_k_ratio=init_k_ratio)
                    module.register_module(f"_learnable_k", lk)
                    _LEARNABLE_K[name] = lk
                h = module.fc2.register_forward_pre_hook(
                    _make_fc2_pre_hook(name, "decoder"), with_kwargs=False
                )
                _HOOK_HANDLES.append(h)
                if not getattr(module, "_grad_hook_installed", False):
                    module.fc1.weight.register_hook(_make_fc1_weight_grad_hook(name))
                    if module.fc1.bias is not None:
                        module.fc1.bias.register_hook(_make_fc1_bias_grad_hook(name))
                    module.fc2.weight.register_hook(_make_fc2_weight_grad_hook(name))
                    module._grad_hook_installed = True
                n_dec += 1

    print(f"[SparseGrad] encoder: {n_enc} layers, mode={enc_mode}, k={enc_k}")
    print(f"[SparseGrad] decoder: {n_dec} layers, mode={dec_mode}, k={dec_k}")
    if learnable_k:
        print(f"[SparseGrad] Learnable k_ratio ENABLED (init={init_k_ratio})")
    if track_wins:
        print(f"[SparseGrad] Win count tracking ENABLED")


def set_sparse_grad_k_ratio(model, k_ratio: float, layer_type: Optional[str] = None):
    """动态调整 k_ratio。layer_type=None 时同时调整 encoder 和 decoder。"""
    global _K_RATIOS
    if layer_type:
        _K_RATIOS[layer_type] = float(k_ratio)
    else:
        _K_RATIOS = {"encoder": float(k_ratio), "decoder": float(k_ratio)}


def set_win_collection(enabled: bool):
    """开启/关闭 win count 收集。"""
    global _COLLECT_WINS
    _COLLECT_WINS = enabled


def get_win_tracker() -> Optional[WinCountTracker]:
    return _TRACKER


def prune_low_winners(model, threshold: float = 0.05):
    """软剪枝：win rate < threshold 的神经元冻结梯度（权重保持原值）。

    前向不封死：pruned 神经元仍参与 WTA top-k 选择。
    若某 pruned 神经元在后续 epoch 被频繁选中（win rate 上升），unfreeze_all 后可恢复训练。

    Args:
        model: WhisperForConditionalGeneration
        threshold: win rate 低于此值的神经元梯度被冻结
    """
    global _GRAD_MASKS

    if _TRACKER is None:
        print("[prune] No win tracker, skipping")
        return 0

    total_pruned = 0
    total_neurons = 0

    for name, module in model.named_modules():
        clsname = module.__class__.__name__
        if clsname not in ("WhisperEncoderLayer", "WhisperDecoderLayer"):
            continue
        if name not in _TRACKER.win_counts:
            continue

        win_rate = _TRACKER.win_rate(name)
        if win_rate is None:
            continue

        d_ff = win_rate.shape[0]
        # 活跃 mask：win rate >= threshold（True = 活跃，False = 冻结梯度）
        active_mask = (win_rate >= threshold)
        n_pruned = int((~active_mask).sum().item())

        # 只保存梯度 mask，不置零权重
        _GRAD_MASKS[name] = active_mask.to(module.fc1.weight.device)

        total_pruned += n_pruned
        total_neurons += d_ff

    pct = total_pruned / max(1, total_neurons) * 100
    print(f"[prune] Soft-pruned {total_pruned}/{total_neurons} neurons ({pct:.1f}%) — gradients frozen, weights preserved")
    return total_pruned


def unfreeze_all():
    """清空所有梯度 mask，允许全部神经元恢复训练（复活机制）。"""
    global _GRAD_MASKS
    n = len(_GRAD_MASKS)
    _GRAD_MASKS = {}
    print(f"[prune] Cleared {n} grad masks — all neurons back to training")


def reset_win_counts():
    """重置 win count 统计。"""
    global _TRACKER
    if _TRACKER is not None:
        _TRACKER.reset()


def get_sparsity_loss() -> torch.Tensor:
    """返回可学习 k_ratio 的稀疏损失 = mean(k_ratio) 各层平均。

    最小化此损失 → 更少的神经元激活 → 隐式剪枝。
    在训练脚本中以 λ * sparsity_loss 加入总 loss。
    """
    if not _LEARNABLE_K:
        return torch.tensor(0.0, requires_grad=False)
    ratios = [torch.sigmoid(lk.logit_k) for lk in _LEARNABLE_K.values()]
    return torch.stack(ratios).mean()


def remove_sparse_grad(model=None):
    """移除所有注册的 pre-forward hooks 和梯度 mask。"""
    global _HOOK_HANDLES, _GRAD_MASKS
    for h in _HOOK_HANDLES:
        h.remove()
    _HOOK_HANDLES = []
    _GRAD_MASKS = {}


# ═══════════════════════════════════════════════════════════════
# 冒烟测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.environ.setdefault("HF_HOME", r"F:\hf_cache")
    os.environ.setdefault("TORCH_HOME", r"F:\hf_cache")

    from transformers import WhisperForConditionalGeneration

    print("Loading whisper-small ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", attn_implementation="eager",
    ).eval()

    print("Applying: encoder=WTA, decoder=SparseGrad, track_wins=True ...")
    apply_sparse_grad_wta(
        model,
        target="encoder+decoder",
        encoder_mode="wta_activation",
        decoder_mode="sparse_grad",
        encoder_k_ratio=0.5,
        decoder_k_ratio=0.3,
        track_wins=True,
    )

    dummy_features = torch.randn(1, 80, 3000)
    dummy_ids = torch.randint(0, model.config.vocab_size, (1, 10))
    with torch.no_grad():
        out = model(input_features=dummy_features, decoder_input_ids=dummy_ids)
    print("Forward OK. logits shape:", out.logits.shape)

    # Check win tracker
    tracker = get_win_tracker()
    if tracker:
        for name, counts in tracker.win_counts.items():
            print(f"  {name}: {counts.sum().item()} wins, {len(counts)} neurons")

    remove_sparse_grad(model)
    print("Removed patch.")
