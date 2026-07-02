# 点心杯粤语 ASR 项目 — τ-dux

基于 `openai/whisper-small` 微调的粤语语音识别模型，参加"点心杯"粤语 ASR 竞赛。项目探索了 τ-opus 可学习注意力归一化、WTA 稀疏激活与 SparseGrad 梯度稀疏等技术的联合效果，在粤语万句多用途生活场景语料集上实现 CER 9.96%。

## 技术方法

### τ-opus 可学习注意力归一化

标准 attention 使用 softmax 归一化：

```
P_ij = exp(s_ij) / Σ_k exp(s_ik)
```

其中 `s_ij = Q_i · K_j / √d` 是注意力分数。τ-opus 将其替换为更具表达力的形式：

```
σ(s) = sigma_function(s)               # per-type 配置
P_ij = σ(s_ij)^τ / Σ_k σ(s_ik)^τ      # τ 控制聚焦度
```

τ 控制注意力分布的锐度：

- **τ = 1**：当 σ=exp 时退化为标准 softmax
- **τ > 1**：注意力分布锐化，聚焦更强
- **τ → 0**：注意力分布平滑，接近均匀

#### 为什么用 σ(s)^τ 替代 softmax

标准 softmax 的指数函数 exp(s) 在正数区间增长过快，导致注意力分布容易"赢家通吃"；同时其导数特性使得梯度更新在极端值区域不稳定。τ-opus 通过选择不同的 σ 函数解耦了两个功能：

1. **σ 函数**：将分数映射到合适的值域，控制分布的形状族
2. **τ 指数**：在分布形状内调节聚焦度，一个标量即可控制锐度

这种分解使得 τ 可以独立于 σ 的类型进行学习和调节。

#### Per-head 可学习 τ

whisper-small 共有 36 个 attention module（12 encoder_self + 12 decoder_self + 12 decoder_cross），每层 12 heads，总计 **432 个 attention heads**。

每个 head 有两个可学习参数：

- **`log_tau`**：`τ = softplus(log_tau) + 1.0`，保证 τ > 1
- **`log_alpha`**：`α = exp(log_alpha)`，shape 调节因子

τ 在训练过程中自由优化，不加强制正则。

#### Per-type σ 函数配置

不同 attention 类型使用不同的 σ 函数：

| attention 类型 | σ 函数 | 原因 |
|---------------|--------|------|
| encoder_self | `softplus` | 连续可微，梯度为 sigmoid，缓解梯度消失 |
| decoder_self | `sigmoid` | 输出限制在 (0,1)，增强自回归稳定性 |
| decoder_cross | `exp` | τ=1 时退化为 softmax，保持与预训练分布一致 |

#### τ* 初始化策略

τ* 估计器为每个 attention head 找到使 KL 散度最小的初始 τ 值：

```
τ* = argmin_τ D_KL(P_τ(s) || P_true(s))
```

使用 Newton 迭代（30 步）求解，以 `Cov(s, log σ(s)) / Var(log σ(s))` 作为初始猜测。

在 ERP2 阶段，基于 ERP1 权重在 5 个随机子集上分别估计 τ*，每头取中位数聚合。关键发现：

- **encoder_self**：τ* 分布最广（1.31-10.00），不同 head/层对聚焦度需求差异大
- **decoder_self**：τ* 均值 2.94，略高于 encoder，自回归需要更强聚焦
- **decoder_cross**：τ* 接近 1.0，softmax 已接近最优

τ* 仅用于初始化可学习 τ，训练中自由优化。

### WTA 稀疏激活

Winner-Take-All (WTA) 稀疏激活作用于 FFN 的前向传播。在 FFN 中间层（4d 维度），只保留 top-k 个激活神经元，其余置零：

```
h = ReLU(W_up · x)
h_sparse = top_k(h, k)             # 保留 top-k 激活
output = W_down · h_sparse
```

k 通过课程学习（curriculum learning）逐步降低：

| Epoch | encoder k_ratio | decoder k_ratio |
|-------|----------------|----------------|
| 1 | 0.60 | 0.50 |
| 2 | 0.55 | 0.45 |
| 3 | 0.50 | 0.40 |
| 4 | 0.45 | 0.35 |
| 5 | 0.40 | 0.30 |

### SparseGrad 梯度稀疏

SparseGrad 在反向传播中只更新 top-k 赢家神经元对应的梯度，其余梯度置零。与 WTA 共享同一 mask（前向赢家 = 反向赢家），实现前向 + 反向的联合稀疏：

```
# 前向：WTA 选择
mask = top_k_mask(h, k)
h_sparse = h * mask

# 反向：SparseGrad 零梯度
∂L/∂h = ∂L/∂h_sparse * mask      # 非赢家梯度为 0
```

效果：参数仅选择性适应粤语数据，隐式正则化，防止小数据过拟合。

#### 动态剪枝

epoch 3 起，对 win rate < 5% 的神经元执行软剪枝——梯度冻结但权重保留，前向仍参与 WTA top-k 选择，允许自然复活。

### Flash τ-opus Triton Kernel

为支持 ERP3 全参微调（241M 参数、全序列长度 1500）下的高效训练，使用 Triton 实现 FlashAttention 风格的 τ-opus 融合算子（`stau_opus_flash.py`）：

**前向融合**：`Q@K^T` → `σ^τ` 归一化 → `attn@V` 在单个 Triton kernel 中完成，避免物化完整的 score 矩阵 `(B, H, L, L)` 到 HBM。

**反向传播**：使用双 kernel 策略

- **dQ kernel**：按 Q-block 遍历 KV
- **dK/dV kernel**：按 KV-block 遍历 Q

反向传播的核心梯度公式：

```
P_ij = σ(s_ij)^τ / Σ_k σ(s_ik)^τ
weighted_i = dO_i · O_i

F_ij = τ · P_ij · (dP_ij - weighted_i) · σ'(s_ij) / σ(s_ij) · α

dQ_i = Σ_j F_ij · K_j - total_F_i · K[argmax_i]
dK_j = Σ_i F_ij · Q_i - total_F_j · Q[j_at_argmax]
dV_j = Σ_i P_ij · dO_i
```

**argmax correction**：在 PyTorch 中做全局 argmax correction，避免 per-block 近似误差。

**精度**：所有 `tl.dot` 使用 `input_precision="ieee"`（完整 float32），不引入 tf32 精度损失。

## 训练流程（三阶段）

项目采用三阶段流水线策略：

| 阶段 | 方法 | 训练范围 | 目标 | CER |
|------|------|---------|------|-----|
| ERP1 | Decoder-only LoRA 微调 | decoder 仅（encoder 冻结） | 快速适应粤语数据 | 9.96% |
| ERP2 | τ* 注意力温度估计 | 仅推理，不训练 | 计算 per-head τ* 用于初始化 | — |
| ERP3 | 全参微调 + τ-opus + WTA + 剪枝 | encoder + decoder（241M） | 最终模型 | 进行中 |

### ERP1: Base 适应

- 训练范围：decoder-only（encoder 冻结）
- Attention：标准 sdpa（不开 τ-opus）
- 正则：SparseGrad k=0.3 + EMA（衰减 0.999）
- 精度：bf16
- Epoch 1 CER=10.62%，Epoch 2 CER=**9.96%**（best）
- 产物：462 MB checkpoint

### ERP2: τ* 估计

- 基于 ERP1 best_model
- σ：encoder_self=softplus, decoder_self=sigmoid, decoder_cross=exp
- Newton 30 步 × 5 次随机子集，中位数聚合
- 产物：432 heads 的 τ* 分布（`erp2_tau_star.json`）

### ERP3: 最终训练（进行中）

- 全参微调（encoder + decoder），241M 参数
- 初始化：ERP1 best_model + ERP2 τ* per-head 初始化
- τ-opus：全部 attention 替换为可学习 τ-opus
- WTA + SparseGrad：课程学习，k 从 0.6/0.5 线性降至 0.4/0.3
- 动态剪枝：epoch 3 起软剪枝
- 数据增强：时序反转（p=0.2）+ MixSpeech（p=0.1）
- 学习率调度：三阶段 LambdaLR（快退火 → 慢退火 → warm restart）
- FlashAttention 风格 Triton fusion kernel
- 状态：2026-07-02 启动训练

## 结果

| 指标 | 值 |
|------|-----|
| 基座模型 | openai/whisper-small |
| 参数量 | 241M |
| ERP1 best CER | **9.96%** |
| ERP3 训练状态 | 进行中 |
| 评测指标 | CER（字错误率）|
| 数据集 | 粤语万句多用途生活场景有声语料集 |

## 使用方式

```bash
# 推理
python predict.py --input_dir ./test_audio --model model.pt --output result.jsonl

# 评估
python evaluator.py result.jsonl verify.csv ./report
```

## 依赖

见 `requirements.txt`。核心依赖：

- PyTorch 2.x
- transformers
- openai-whisper
- triton (Flash τ-opus kernel)
- datasets / soundfile

## 仓库结构

```
├── predict.py                   # 推理脚本
├── model.pt                     # 微调权重
├── requirements.txt             # 依赖
├── scripts/
│   ├── train_erp1.py            # ERP1 训练脚本
│   ├── train_erp3.py            # ERP3 训练脚本
│   ├── patch_whisper_stau.py    # τ-opus 接入 whisper eager attention forward
│   ├── patch_whisper_sparse_grad.py  # WTA + SparseGrad + 剪枝实现
│   ├── stau_opus_operator.py    # τ-opus 算子（旧版，历史参考）
│   ├── stau_opus_flash.py       # Flash τ-opus Triton kernel（当前使用）
│   └── remote.py                # 远程 GPU 服务器管理
├── docs/
│   ├── tau_opus_whisper_implementation.md  # τ-opus 技术文档
│   ├── training_plan_erp_v2.md             # 三阶段训练规划
│   ├── progress_erp1.md                    # ERP1 进度记录
│   ├── progress_erp2.md                    # ERP2 进度记录
│   └── progress_erp3.md                    # ERP3 进度记录
├── data/
│   └── life-scenarios/          # 初赛数据集
└── outputs/
    └── whisper_small_tau_star.json         # τ* 计算结果
```

## 赛事信息

- 赛事：点心杯粤语 ASR 竞赛
- 初赛数据集：粤语万句多用途生活场景有声语料集
- 评测指标：CER（字错误率），越低越好
- 基座模型：`openai/whisper-small`
- 输出形式：粤语汉字文本
