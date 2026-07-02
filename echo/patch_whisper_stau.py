"""
把 τ-opus 注意力归一化接入 Whisper-small。
支持：固定 τ*（来自 tau_star 估计器）或可学习 τ（STauOpusLearnable）。
本文件自包含，不依赖外部 F:\τ\τopus 路径，可直接上传到远程服务器运行。
"""
import os
import sys
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.whisper import modeling_whisper as whisper_module

# Import the fused τ-opus attention operator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stau_opus_operator import STauOpusAttentionFunction

# Flash V2 kernel (optional, for use_flash=True)
try:
    from stau_opus_flash_v2 import STauOpusFlashV2Module, _HAS_TRITON as _HAS_FLASH_V2
except ImportError:
    STauOpusFlashV2Module = None
    _HAS_FLASH_V2 = False

_ORIGINAL_EAGER_FORWARD = whisper_module.eager_attention_forward


# ═══════════════════════════════════════════════════════════════
# τ-opus core: sigma registry + max-stable autograd Function
# ═══════════════════════════════════════════════════════════════

CLAMP_MIN = 1e-8


def _sigma_softplus(x): return F.softplus(x).clamp(min=CLAMP_MIN)
def _sigma_sigmoid(x):  return torch.sigmoid(x).clamp(min=CLAMP_MIN)
def _sigma_exp(x):      return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)
def _sigma_relu(x):     return F.relu(x).clamp(min=CLAMP_MIN)
def _sigma_tanh_shift(x): return (torch.tanh(x) + 1).clamp(min=CLAMP_MIN)


_SIGMA_REGISTRY = {
    "softplus": _sigma_softplus,
    "sigmoid": _sigma_sigmoid,
    "exp": _sigma_exp,
    "relu": _sigma_relu,
    "tanh_shift": _sigma_tanh_shift,
}


def _sigma_prime_softplus(x): return torch.sigmoid(x)
def _sigma_prime_sigmoid(x):
    s = torch.sigmoid(x)
    return s * (1 - s)
def _sigma_prime_exp(x):      return torch.exp(x.clamp(max=15))
def _sigma_prime_tanh(x):     return 1 - torch.tanh(x) ** 2
def _sigma_prime_relu(x):     return (x > 0).float()


_SIGMA_PRIME_REGISTRY = {
    "softplus": _sigma_prime_softplus,
    "sigmoid": _sigma_prime_sigmoid,
    "exp": _sigma_prime_exp,
    "tanh_shift": _sigma_prime_tanh,
    "relu": _sigma_prime_relu,
}


class STauOpusMaxStableFn(torch.autograd.Function):
    """Max-stabilized τ-opus: σ(x - max)^τ / Σσ(x - max)^τ。"""
    @staticmethod
    def forward(ctx, scores, tau, sigma_name):
        sigma_fn = _SIGMA_REGISTRY[sigma_name]
        x_stable = scores - scores.max(dim=-1, keepdim=True).values
        sigma_val = sigma_fn(x_stable)
        q = sigma_val.clamp(min=CLAMP_MIN).pow(tau)
        sum_q = q.sum(dim=-1, keepdim=True).clamp(min=CLAMP_MIN)
        attn = q / sum_q
        ctx.save_for_backward(scores, x_stable, sigma_val, attn, sum_q)
        ctx.tau = tau
        ctx.sigma_name = sigma_name
        return attn

    @staticmethod
    def backward(ctx, dO):
        scores, x_stable, sigma_val, attn, sum_q = ctx.saved_tensors
        tau, sigma_name = ctx.tau, ctx.sigma_name
        sigma_prime_fn = _SIGMA_PRIME_REGISTRY.get(sigma_name, lambda x: torch.ones_like(x))

        sigma_tau_m1 = sigma_val.clamp(min=CLAMP_MIN).pow(tau - 1.0)
        sigma_p = sigma_prime_fn(x_stable)
        A = tau * sigma_tau_m1 * sigma_p
        S = sum_q
        q = sigma_val.clamp(min=CLAMP_MIN).pow(tau)

        wda = (dO * q).sum(dim=-1, keepdim=True)
        da = (dO * A).sum(dim=-1, keepdim=True)
        sA = A.sum(dim=-1, keepdim=True)

        term1 = A * (dO / S - wda / S.pow(2))
        term2_full = da / S - wda * sA / S.pow(2)
        argmax = scores.argmax(dim=-1, keepdim=True)
        term2 = torch.zeros_like(term1)
        term2.scatter_(-1, argmax, -term2_full)

        dScores = term1 + term2
        return dScores, None, None


class STauOpusLearnable(nn.Module):
    """τ-opus with learnable τ, α per head (fused autograd).

    σ(s) = softplus(α·s), α = exp(log_α) > 0
    τ = softplus(log_τ) + 1.0, τ > 1

    Uses STauOpusAttentionFunction for custom backward (faster, less memory).
    """
    def __init__(self, n_heads, sigma_name="softplus"):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(n_heads))
        self.log_alpha = nn.Parameter(torch.zeros(n_heads))
        self.sigma_name = sigma_name

    def forward(self, Q, K, V, attn_mask=None, scale=None):
        out, _ = STauOpusAttentionFunction.apply(
            Q, K, V, self.log_tau, self.log_alpha, self.sigma_name, attn_mask, scale
        )
        return out


# ═══════════════════════════════════════════════════════════════
# Whisper attention classification & tau dict loading
# ═══════════════════════════════════════════════════════════════

def _classify_attn(name: str, module):
    """根据模块名判断是 encoder self / decoder self / decoder cross。"""
    is_decoder = getattr(module, "is_decoder", False)
    if not is_decoder:
        return "encoder_self"
    if "encoder_attn" in name:
        return "decoder_cross"
    return "decoder_self"


def load_tau_dict(json_path: Union[str, Path]) -> Dict[str, Dict[int, list]]:
    """
    读取 tau_star 估计器输出的 JSON，返回
    {"encoder_self": {layer_idx: [H tau values], ...}, ...}

    兼容两种格式：
      - compute_whisper_tau_star.py: {"is_decoder", "cross_attention", "layer_idx"}
      - compute_tau_star_iop.py: {"module_name": "model.encoder.layers.0.self_attn", ...}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {"encoder_self": {}, "decoder_self": {}, "decoder_cross": {}}
    for item in data:
        if "module_name" in item:
            name = item["module_name"]
            if "encoder_attn" in name:
                key = "decoder_cross"
            elif "decoder" in name and "self_attn" in name:
                key = "decoder_self"
            else:
                key = "encoder_self"
            layer = _parse_layer_idx(name)
        else:
            attn_type = item.get("attention_type")
            if attn_type and attn_type in out:
                key = attn_type
            else:
                key = "decoder_cross" if item.get("cross_attention") else (
                    "decoder_self" if item.get("is_decoder") else "encoder_self")
            layer = item.get("layer_idx")
            if layer is None:
                layer = 0
        out[key][layer] = item["tau_per_head"]
    return out


# ═══════════════════════════════════════════════════════════════
# Forward patches
# ═══════════════════════════════════════════════════════════════

def _make_fixed_stau_forward():
    def stau_eager_attention_forward(
        module, query, key, value, attention_mask,
        scaling=None, dropout=0.0, **kwargs
    ):
        if scaling is None:
            scaling = query.size(-1) ** -0.5

        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        tau = getattr(module, "_stau_tau", None)
        if tau is None:
            tau = 1.0
        else:
            tau = tau.to(scores.device)

        sigma_name = getattr(module, "_stau_sigma", "softplus")
        attn_weights = STauOpusMaxStableFn.apply(scores.float(), tau, sigma_name)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights.to(value.dtype), value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights.to(value.dtype)
    return stau_eager_attention_forward


def _make_learnable_stau_forward():
    """使用 STauOpusLearnable（融合算子），τ 和 α 可学习。"""
    def stau_eager_attention_forward(
        module, query, key, value, attention_mask,
        scaling=None, dropout=0.0, **kwargs
    ):
        opus = getattr(module, "_stau_opus", None)
        if opus is None:
            raise RuntimeError("Module has no _stau_opus. Call apply_learnable_stau first.")

        if scaling is None:
            scaling = query.size(-1) ** -0.5

        attn_output = opus(query, key, value, attn_mask=attention_mask, scale=scaling)
        attn_output = F.dropout(attn_output, p=dropout, training=module.training)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    return stau_eager_attention_forward


def _make_learnable_stau_forward_flash():
    """使用 STauOpusFlashV2Module（Triton Flash kernel），τ 和 α 可学习。
    
    优势：
    - 避免材料化 (B,H,Lq,Lk) score 矩阵到 HBM
    - backward 中 dτ/dα 在 Triton kernel 内累加，无 per-head Python loop
    - 预计比 eager 版快 2-3x
    """
    def stau_flash_attention_forward(
        module, query, key, value, attention_mask,
        scaling=None, dropout=0.0, **kwargs
    ):
        opus = getattr(module, "_stau_opus", None)
        if opus is None:
            raise RuntimeError("Module has no _stau_opus. Call apply_learnable_stau first.")

        if scaling is None:
            scaling = query.size(-1) ** -0.5

        attn_output = opus(query, key, value, attn_mask=attention_mask, scale=scaling)
        attn_output = F.dropout(attn_output, p=dropout, training=module.training)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    return stau_flash_attention_forward


# ═══════════════════════════════════════════════════════════════
# Apply / remove patches
# ═══════════════════════════════════════════════════════════════

def _log_tau_for_init(tau: float) -> float:
    """STauOpusLearnable: tau = softplus(log_tau) + 1.0。返回使初始 tau≈target 的 log_tau。"""
    if tau <= 1.001:
        return -8.0
    return math.log(math.exp(tau - 1.0) - 1.0)


def _parse_layer_idx(name: str) -> int:
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else 0


def apply_fixed_stau(model, tau_source, sigma_name="softplus"):
    """
    给 model 的每个 WhisperAttention 模块绑定固定 per-head τ。

    Args:
        model: WhisperForConditionalGeneration
        tau_source:
            - float / int：所有 head 用同一个 τ。
            - dict: {"encoder_self": {layer_idx: [H floats]}, ...}
            - str / Path: tau_star 估计器输出的 JSON 路径
        sigma_name: str 或 dict，分别指定 encoder/decoder/cross 的 σ 函数。
    """
    if isinstance(tau_source, (str, Path)):
        tau_dict = load_tau_dict(tau_source)
    elif isinstance(tau_source, dict):
        tau_dict = tau_source
    elif isinstance(tau_source, (float, int)):
        tau_dict = None
        scalar_tau = float(tau_source)
    else:
        raise TypeError(f"Unsupported tau_source type: {type(tau_source)}")

    if isinstance(sigma_name, str):
        sigma_map = {k: sigma_name for k in ("encoder_self", "decoder_self", "decoder_cross")}
    elif isinstance(sigma_name, dict):
        sigma_map = {k: sigma_name.get(k, "softplus") for k in ("encoder_self", "decoder_self", "decoder_cross")}
    else:
        raise TypeError(f"Unsupported sigma_name type: {type(sigma_name)}")

    for name, module in model.named_modules():
        if module.__class__.__name__ != "WhisperAttention":
            continue
        h = module.num_heads
        kind = _classify_attn(name, module)
        if tau_dict is not None:
            layer = _parse_layer_idx(name)
            tau_list = tau_dict.get(kind, {}).get(layer, [1.0] * h)
            tau_tensor = torch.tensor(tau_list, dtype=torch.float32).view(1, h, 1, 1)
        else:
            tau_tensor = torch.full((1, h, 1, 1), scalar_tau, dtype=torch.float32)
        module._stau_tau = tau_tensor
        module._stau_sigma = sigma_map[kind]

    whisper_module.eager_attention_forward = _make_fixed_stau_forward()


def apply_learnable_stau(
    model,
    tau_init: Union[float, str, Path, Dict[str, Dict[int, list]]] = 1.0,
    alpha_init: float = 1.0,
    sigma_name: Union[str, Dict[str, str]] = "softplus",
    default_tau_json: Union[str, Path] = r"F:\τ\点心杯\outputs\base_iop_tau_star.json",
    use_flash: bool = False,
):
    """给每个 WhisperAttention 挂一个 STauOpus 算子，τ 和 α 可学习。

    Args:
        tau_init:
            - float: 所有 head 初始化为该 τ（实际通过 softplus+1 映射）。
            - "tau_star" 或 json path: 从 tau_star JSON 按层/头加载 per-head 初始化。
            - dict: {"encoder_self": {layer_idx: [H floats], ...}, ...}
        alpha_init: 所有 head 的初始 α。
        sigma_name: str 或 dict。str 时所有 attention 用同一 σ；
                    dict 时按 encoder_self/decoder_self/decoder_cross 分别配置。
        default_tau_json: tau_init="tau_star" 时使用的默认 JSON 路径。
        use_flash: 使用 Flash V2 Triton kernel（更快，但需要 Triton + CUDA）。
    """
    if isinstance(tau_init, (str, Path)):
        if str(tau_init).lower() == "tau_star":
            tau_dict = load_tau_dict(default_tau_json)
        else:
            tau_dict = load_tau_dict(tau_init)
    elif isinstance(tau_init, dict):
        tau_dict = tau_init
    elif isinstance(tau_init, (float, int)):
        tau_dict = None
        scalar_tau = float(tau_init)
    else:
        raise TypeError(f"Unsupported tau_init type: {type(tau_init)}")

    if isinstance(sigma_name, str):
        sigma_map = {k: sigma_name for k in ("encoder_self", "decoder_self", "decoder_cross")}
    elif isinstance(sigma_name, dict):
        sigma_map = {k: sigma_name.get(k, "softplus") for k in ("encoder_self", "decoder_self", "decoder_cross")}
    else:
        raise TypeError(f"Unsupported sigma_name type: {type(sigma_name)}")

    # Resolve flash mode
    actual_flash = use_flash and _HAS_FLASH_V2 and torch.cuda.is_available()
    if use_flash and not actual_flash:
        print(f"[τ-opus] ⚠ Flash V2 requested but unavailable (Triton={_HAS_FLASH_V2}, CUDA={torch.cuda.is_available()})")
        print(f"[τ-opus]   Falling back to eager STauOpusLearnable")

    # Choose module class
    ModuleClass = STauOpusFlashV2Module if actual_flash else STauOpusLearnable

    n_attached = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "WhisperAttention":
            continue
        h = module.num_heads
        kind = _classify_attn(name, module)

        opus = ModuleClass(h, sigma_name=sigma_map[kind]).to(next(module.parameters()).dtype)
        opus.log_alpha.data.fill_(math.log(alpha_init))

        if tau_dict is not None:
            layer = _parse_layer_idx(name)
            tau_list = tau_dict.get(kind, {}).get(layer, [1.0] * h)
        else:
            tau_list = [scalar_tau] * h

        tau_list = [max(1.0, float(t)) for t in tau_list[:h]]
        opus.log_tau.data = torch.tensor(
            [_log_tau_for_init(t) for t in tau_list],
            dtype=opus.log_tau.dtype, device=opus.log_tau.device,
        )
        module._stau_opus = opus
        n_attached += 1

    backend_name = "Flash V2 (Triton)" if actual_flash else "Eager (PyTorch)"
    print(f"[τ-opus] Attached {backend_name} STauOpus to {n_attached} attention modules")
    print(f"  σ config: {sigma_map}")
    if actual_flash:
        whisper_module.eager_attention_forward = _make_learnable_stau_forward_flash()
    else:
        whisper_module.eager_attention_forward = _make_learnable_stau_forward()


def remove_stau(model=None):
    """恢复原始 eager_attention_forward，并清理附加属性。"""
    whisper_module.eager_attention_forward = _ORIGINAL_EAGER_FORWARD
    if model is not None:
        for module in model.modules():
            if module.__class__.__name__ == "WhisperAttention":
                module._stau_tau = None
                module._stau_opus = None


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.environ.setdefault("HF_HOME", r"F:\hf_cache")
    os.environ.setdefault("TORCH_HOME", r"F:\hf_cache")

    from transformers import WhisperForConditionalGeneration

    print("Loading whisper-small ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        attn_implementation="eager",
    ).eval()

    print("Applying fixed τ=1.0 τ-opus attention (smoke test) ...")
    apply_fixed_stau(model, tau_source=1.0, sigma_name="softplus")

    dummy_features = torch.randn(1, 80, 150)
    dummy_ids = torch.randint(0, model.config.vocab_size, (1, 10))
    with torch.no_grad():
        out = model(input_features=dummy_features, decoder_input_ids=dummy_ids)
    print("Forward OK. logits shape:", out.logits.shape)

    remove_stau(model)
    print("Removed stau patch.")
