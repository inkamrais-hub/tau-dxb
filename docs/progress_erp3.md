# ERP3 训练进度记录

## 概览

| 项目 | 内容 |
|------|------|
| 阶段 | erp v2 阶段 3 |
| 目标 | 全参微调 + τ-opus + WTA 前向稀疏 + 剪枝 + SparseGrad + 数据增强 |
| 训练范围 | 全参微调（encoder + decoder），241M 参数 |
| 状态 | 🔄 训练中（2026-07-02 11:04 启动） |

## 配置

| 参数 | 值 |
|------|-----|
| 基座 | `openai/whisper-small` |
| 训练范围 | 全参微调（encoder + decoder） |
| 初始化 | ERP1 best_model + ERP2 τ* 初始化 |
| Epoch | 5 |
| Effective batch | 16（BATCH=4, GRAD_ACCUM=4） |
| LR (base) | 3e-5 |
| LR (τ/α) | 1e-3 |
| Scheduler | LambdaLR 三阶段：epoch 1-2 快退火 / epoch 3-4 慢退火 / epoch 5 warm restart → 0 |
| Precision | bf16 |
| Gradient clipping | 1.0 |
| EMA | 启用，衰减 0.999 |

### 组件分配

| 组件 | encoder | decoder |
|------|---------|---------|
| FFN 前向 | WTA 激活稀疏（top-k） | 全量 |
| FFN 反向 | WTA 梯度稀疏（同一 mask） | SparseGrad 梯度稀疏 |
| 剪枝 | 有（第 3 epoch 起） | 无 |
| attention | τ-opus 可学习（softplus） | τ-opus 可学习（self=sigmoid, cross=exp） |

### τ-opus 可学习配置

- 36 个 attention module 各挂一个 `STauOpusLearnable`
- σ 分类型：encoder_self=softplus, decoder_self=sigmoid, decoder_cross=exp
- τ 初始化为 ERP2 τ*（per-head），训练中自由优化

### WTA/SparseGrad 课程

| Epoch | encoder k_ratio | decoder k_ratio |
|-------|----------------|----------------|
| 1 | 0.60 | 0.50 |
| 2 | 0.55 | 0.45 |
| 3 | 0.50 | 0.40 |
| 4 | 0.45 | 0.35 |
| 5 | 0.40 | 0.30 |

### 剪枝策略

- Epoch 1-2：只统计 win count，不剪枝
- Epoch 3+：每 epoch 末剪枝（win rate < 5% 的神经元置零+冻结梯度）
- 剪枝为软剪枝：梯度冻结但权重保留，前向仍参与 WTA top-k 选择，允许自然复活

### 数据增强

| 增强 | 概率 | 说明 |
|------|------|------|
| 时序反转 | 0.2 | 只反转音频，文本不变 |
| MixSpeech（硬标签） | 0.1 | 音频按 9:1 混合，标签用主样本 |

### Flash Attention Kernel

ERP3 使用 FlashAttention 风格的 τ-opus 融合 Triton kernel：

- **前向**：`Q@K^T` → `σ^τ` 归一化 → `attn@V` 全部 fusion 进单个 kernel
- **反向**：dQ/dK/dV 各用一个 Triton kernel，使用全局 `weighted = dO·O` 避免 per-block 近似误差
- **精度**：所有 `tl.dot` 使用 `input_precision="ieee"`（完整 float32）
- **argmax correction**：在 PyTorch 中做全局 argmax correction
- 支持 Q/K 不同序列长度（decoder cross-attention 中 Lq=29, Lk=1500）

## 训练进度

**启动时间**：2026-07-02 11:04 CST

### 当前状态（Epoch 1）

| 指标 | 值 |
|------|-----|
| 当前 step | ~64 / 2113 |
| 当前 loss | ~4.80 |
| 当前 LR | 2.96e-05 |
| 速度 | ~3.08 s/it（Triton 编译中，逐步加速） |
| encoder k | 0.60 |
| decoder k | 0.50 |

> 首步 ~33s 主要受 Triton kernel 编译影响，后续逐步降至 3s/it。

## 产物

训练中结果将输出到：
- `/hy-tmp/dimsum/outputs/erp3/best_model/model.pt`
- `/hy-tmp/dimsum/outputs/erp3/epoch{1..5}/model.pt`
- `/hy-tmp/dimsum/outputs/erp3/tau_epoch{1..5}.json`
- 训练日志：`/hy-tmp/dimsum/erp3.log`
