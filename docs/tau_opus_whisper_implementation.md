# τ-opus 注意力归一化在 whisper-small 上的实现与训练策略

## 1. τ-opus 原理

### 1.1 标准 Softmax Attention

标准 attention 使用 softmax 归一化：

```
P_ij = exp(s_ij) / Σ_k exp(s_ik)
```

其中 `s_ij = Q_i · K_j / √d` 是注意力分数。

### 1.2 τ-opus 归一化

τ-opus 用 `σ(s)^τ / Σσ(s)^τ` 替换 softmax：

```
σ(s) = sigma_function(s)           # softplus / sigmoid / exp
P_ij = σ(s_ij)^τ / Σ_k σ(s_ik)^τ  # τ 控制聚焦度
```

- **τ = 1**：退化为标准 softmax（当 σ=exp 时）
- **τ > 1**：注意力分布更锐化（聚焦更强）
- **τ → 0**：注意力分布更平滑（接近均匀）

### 1.3 σ 函数的注意力类型配置

| attention 类型 | σ 函数 | 原因 |
|---------------|--------|------|
| encoder_self | `softplus` | 连续可微，允许梯度平滑变化；梯度为 sigmoid，缓解梯度消失 |
| decoder_self | `sigmoid` | 输出限制在 (0,1)，增强自回归稳定性 |
| decoder_cross | `exp` | τ=1 时退化为 softmax，保持与预训练分布一致 |

### 1.4 τ* 最优 τ 估计

给定固定的分数分布 s，τ* 是使以下目标最优的 τ：

```
τ* = argmin_τ D_KL(P_τ(s) || P_true(s))
```

估计方法：Newton 迭代求解（10-30 步），使用 `Cov(s, log σ(s)) / Var(log σ(s))` 作为初始猜测。

## 2. 在 whisper-small 上的落地

### 2.1 架构

whisper-small 有 36 个 attention module：
- 12 × encoder_self_attn（每层 12 heads）
- 12 × decoder_self_attn（每层 12 heads）
- 12 × decoder_cross_attn（每层 12 heads）
- 共 **432 个 attention heads**（12 heads × 12 layers × 3 types）

### 2.2 算子实现演化

#### 版本 1：纯 PyTorch 自定义 backward（已弃用）
- 文件：`stau_opus_triton.py`（旧版 Triton 实现）
- 每个 forward/backward 完整物化 score 矩阵 `(B, H, L, L)` → HBM 开销大

#### 版本 2：融合算子（被取代）
- 文件：`stau_opus_operator.py`
- 自定义 autograd Function，backward 显式求导
- 直接用于 ERP1/ERP2 阶段的验证

#### 版本 3：FlashAttention 风格 Triton 融合算子（当前）
- 文件：`stau_opus_flash.py`
- **前向融合**：`Q@K^T` → `σ^τ` 归一化 → `attn@V` 在单个 Triton kernel 中完成
- **不物化 score 矩阵** `(B, H, L, L)` 到 HBM
- **存储归一化常数**：m（row max）、l（row sum）、argmax（行最大位置）用于反向
- **反向双 kernel**：dQ kernel（按 Q-block 遍历 KV）+ dK/dV kernel（按 KV-block 遍历 Q）
- **全局 weighted**：使用前向输出 O 计算 `weighted = dO·O` 替代 per-block 近似
- **ieee 精度**：所有 `tl.dot` 使用 `input_precision="ieee"` 避免 tf32 精度损失

### 2.3 可学习参数

每个 attention head 有两个可学习参数：
- **`log_tau`**：`τ = softplus(log_tau) + 1.0`，τ > 1
- **`log_alpha`**：`α = exp(log_alpha)`，α > 0

初始 τ 来自 ERP2 的 τ* 估计（per-head），训练中自由优化。

### 2.4 分类型 σ 配置

σ 函数在 `patch_whisper_stau.py` 中按 attention 类型指定：

```python
SIGMA_MAP = {
    "encoder_self": "softplus",
    "decoder_self": "sigmoid", 
    "decoder_cross": "exp",
}
```

每个 `STauOpusLearnable` 模块根据其所属的 attention 类型自动选择对应的 σ 函数。

## 3. ERP3 训练策略

### 3.1 三阶段流水线

#### 阶段 1：ERP1 — Base 适应
- **目标**：快速让 whisper-small 适应粤语生活场景
- **策略**：decoder-only 微调（冻结 encoder），bf16
- **τ-opus**：关闭（使用标准 sdpa attention）
- **正则**：SparseGrad k=0.3 + EMA
- **结果**：CER = 9.96%

#### 阶段 2：ERP2 — τ* 估计
- **目标**：为每个 attention head 估计最优 τ*
- **方法**：基于 ERP1 权重，采样子集数据做 Newton 迭代
- **结果**：36 modules, 432 heads 的 τ* 分布
- **关键发现**：decoder_cross τ* ≈ 1.0（接近 softmax）；encoder_self τ* 分布最广（1.31–10.00）

#### 阶段 3：ERP3 — 最终训练
- **全参微调**：encoder + decoder，241M 参数全部训练
- **τ 可学习**：用 ERP2 τ* 初始化 per-head log_tau
- **WTA 前向稀疏 + SparseGrad 课程**：k 从 0.6/0.5 线性降至 0.4/0.3
- **动态剪枝**：epoch 3 起软剪枝（win rate < 5%），允许自然复活
- **数据增强**：时序反转（p=0.2）+ MixSpeech（p=0.1）
- **学习率调度**：三阶段 LambdaLR

### 3.2 学习率调度设计

```
epoch 1-2: 快周期退火（快速探索 τ 和权重空间）
epoch 3-4: 缓慢退火（精细收敛）
epoch 5:   warm restart + 快速衰减到 0（最终收敛）
```

### 3.3 Flash Attention Kernel 反向传播公式

核心梯度推导：

```
P_ij = σ(s_ij)^τ / Σ_k σ(s_ik)^τ               (前向)
dP_ij = (dO_i · V_j)                             (受注意力的输出梯度)
weighted_i = Σ_j P_ij · dP_ij = dO_i · O_i      (全局加权值)

F_ij = τ · P_ij · (dP_ij - weighted_i) · σ'(s_ij) / σ(s_ij) · α

dQ_i = Σ_j F_ij · K_j - total_F_i · K[argmax_i]  (argmax correction)
dK_j = Σ_i F_ij · Q_i - total_F_j · Q[j_at_argmax]
dV_j = Σ_i P_ij · dO_i
```

## 4. 算子状态与文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `stau_opus_flash.py` | FlashAttention 风格 Triton 融合算子 | ✅ 当前使用 |
| `patch_whisper_stau.py` | 接入 whisper eager attention forward | ✅ 当前使用 |
| `stau_opus_operator.py` | 旧版融合算子 | ⏸️ 历史版本 |
| `patch_whisper_sparse_grad.py` | SparseGrad + WTA + 剪枝实现 | ✅ |
| `train_erp1.py` | ERP1 训练脚本 | ✅ |
| `train_erp3.py` | ERP3 训练脚本 | ✅ |

## 5. 已知限制

1. **Triton kernel 编译开销**：首次调用时 ~30s，后续稳定
2. **ieee 精度降低速度**：完整 float32 精度比 tf32 慢约 10-15%
3. **不支持 attention mask（padding mask）的完整融合**：当前通过 `tl.where` 实现 padding mask
4. **τ-opus 仅在 eager mode 可用**：不支持 transformers 的 `attn_implementation="sdpa"` 或 "flash_attn"
