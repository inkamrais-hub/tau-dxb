# Base-XL-v2 训练策略

## 一、E-v1 问题诊断

| 问题 | 影响 | 修复 |
|------|------|------|
| τ 固定为 buffer | 模型被迫适应 τ*=2~3 的尖锐分布，初始 loss=4.6 | τ → nn.Parameter 可学习 |
| LR=1e-5 过低 | 5 epoch 学不动 | LR → 5e-5 |
| 无数据增强 | 8449 条数据易过拟合 | 时序反转 + MixSpeech |
| 无 τ 课程 | 从头硬扛 τ-opus 的陌生 landscape | τ 初始 1.0，训练中自适应 |

**根因**：τ 固定死是最关键的问题。可学习 τ 让每个 head 自己决定"应该多尖锐"，初始 =1.0(≡softmax)，让模型先正常收敛，再自动找到最优 τ。

---

## 二、训练配置

| 参数 | E-v1 (旧) | E-v2 (新) |
|------|-----------|-----------|
| 学习率 | 1e-5 | **5e-5** |
| Epoch | 5 | **10** |
| τ | buffer 固定 | **nn.Parameter** 可学习，初始 1.0 |
| τ 优化器 | 无 | 与 decoder 一起 AdamW |
| SparseGrad | k=0.3 固定 | k=0.5→0.3 课程 |
| 数据增强 | 无 | 时序反转 50% + MixSpeech α=0.1 |
| 时间估计 | ~30min | ~2h |

### 可学习 τ 的实现

```python
# 每个 attention head 独立 τ
# 类型: nn.Parameter 而非 buffer
# 初始值: 1.0 (等价 softmax)
# 范围约束: clamp(min=0.5, max=5.0)
# 优化器: 加入 AdamW，与 decoder 参数同 LR

module._stau_tau = nn.Parameter(torch.ones(1, num_heads, 1, 1))
```

训练过程中 τ 通过 τ-opus 的反传（`STauOpusFn.backward`）接收梯度，自动调整。

---

## 三、数据增强

### 3.1 时序反转（Time Reversal）

每步 50% 概率翻转音频 + 反转文本标签：

```python
if random.random() < 0.5:
    audio = np.flip(audio).copy()      # 时间反转
    text = text[::-1]                   # "早晨" → "晨早"
```

**效果**：迫使 encoder 不依赖时间顺序 shortcut，学真正的声学特征。

### 3.2 MixSpeech（音频 Mixup）

随机配对两条样本，音频按比例混合、标签按比例平滑：

```python
mix_ratio = 0.1  # 小比例混合
mix_audio = (1 - mix_ratio) * audio_A + mix_ratio * audio_B
# loss = (1 - mix_ratio) * ce(pred, text_A) + mix_ratio * ce(pred, text_B)
```

**注意**：比例小（0.1），防止破坏声学特征。主要目的是正则化。

### 3.3 随机噪声扰动（可选）

```python
if random.random() < 0.3:
    noise = torch.randn_like(audio) * 0.005
    audio = audio + noise
```

---

## 四、τ* 估计器重新校准

### 4.1 流程（已优化）

1. 用 `scripts/compute_tau_star_b.py` 在 **微调后的 B 模型** 上跑 10 次。
2. 每次抽取不同子集（12 条），仅计算 **encoder self-attention** 的 τ*。
3. 每次 forward 直接调用 `model.encoder()`，不走 `generate()`，省掉 decoder 开销。
4. 对每帧随机裁剪到 ≤400 帧，并在每个 head 上只采样 8 个 query 位置和 256 个 key 位置。
5. 用 **Newton 标量迭代** 替代原来的 100 步 SGD，收敛更快、KL 更低。
6. 取 10 次 run 的**中位数**作为稳定 τ* 值。

### 4.2 τ* 的作用

| 用途 | 说明 |
|------|------|
| **初始化参考** | τ 可学习初始化为 1.0，τ* 作为"期望终值"参考 |
| **KV 压缩** | `k_h = max(2, T / τ_h)` — τ* 越高，压缩越狠 |
| **消融分析头** | τ* 异常高/低的 head 可能是冗余或关键 head |

### 4.3 对 τ 估计器的影响

τ* 估计器是基于 **原始 softmax** 计算的（hook pre-softmax scores 后对标 softmax 分布计算最优 τ）。它不依赖训练后的模型状态，所以：

- τ* 估计器 **不受训练影响** — 只要在你仓库的 `τopus/tau_star_opus.py` 能跑，就能出 τ*
- E-v2 中 τ 虽然可学习，但 τ* 仍是初始化参考 + KV 压缩依据
- KV 压缩用 τ* 而非训练的 τ，因为 τ* 是"注意力聚焦度"的度量，压缩策略应该基于模型能力而非训练状态

---

## 五、τ*-KV 压缩方案

### 5.1 核心公式

```python
# 对 encoder 第 l 层第 h 头
τ_h = τ*_l_h                        # 从 τ* 估计器获取
T = sequence_length                 # 当前输入长度
k_h = max(K_MIN, T // τ_h)          # 保留的 KV 帧数

# 推理时
scores = Q @ K^T / sqrt(d)          # 计算全量 score
topk_val, topk_idx = scores.topk(k_h, dim=-1)  # 只取 top-k 帧
# 用 topk_idx 从 K, V 中 gather 对应帧
```

### 5.2 预期效果

| τ* | 注意力行为 | k_h (T=1500) | 压缩率 |
|----|-----------|-------------|--------|
| 1.0 | 平滑 | 1500 | 0% |
| 2.0 | 中等聚焦 | 750 | 50% |
| 3.0 | 尖锐 | 500 | 67% |
| 5.0 | 极尖锐 | 300 | 80% |

### 5.3 精度影响分析

**潜在风险**：
- KV 压缩本质是有损的，丢弃了非 top-k 帧的信息
- 如果某些 head 的 τ* 虽然高但偶尔需要访问非 top-k 帧（长距离依赖），压缩会导致精度下降
- decoder cross-attention 对完整性更敏感（需要看到完整编码输出）

**缓解方案**：
1. **min_keep 下限**：`K_MIN = 64`，保证即使 τ* 极高也有最低帧数
2. **分层压缩**：encoder 层压 50%，decoder cross-attention 压 20% 或不压
3. **A/B 消融**：对比有/无 KV 压缩的 CER

### 5.4 估算

- τ* = 1.0-5.0 范围内，encoder 平均压缩率 ~40%
- 对 whisper-small：KV cache 从 ~1500×384×12 → ~900×384×12
- 显存节省：~40%，推理速度提升：~1.5x
- 预期 CER 增加：<0.5%（需验证）

---

## 六、SparseGrad 课程

```
Epoch 1-2:  k_ratio = 0.5  (50% 神经元保留，梯度充分流动)
Epoch 3-4:  k_ratio = 0.4
Epoch 5+:   k_ratio = 0.3  (30% 收窄，防过拟合)
```

每 epoch 更新一次 `module.k_ratio`。

---

## 七、训练步骤

```
1. 用 compute_whisper_tau_star.py 跑 10 次 τ* 校准 → 输出稳定 τ*.json  ← 当前
2. 用 τ* 初始化 KV 压缩策略
3. 训练 E-v2：
   - τ 可学习，初始 1.0
   - LR=5e-5, epoch=10
   - 数据增强：时序反转 + MixSpeech
   - SparseGrad 课程 0.5→0.3
   - 每 epoch 保存 checkpoint
4. 推理 val 集 + test 集
5. 如果有 KV 压缩：跑一次压缩版推理对比 CER
6. 与 B 的 val CER (8.09%) 对比
```

---

## 八、下一步（这次之后）

| 优先级 | 项目 | 难度 | 预期收益 |
|--------|------|------|---------|
| P0 | τ 可学习 | 低 | **修复根本问题** |
| P1 | 数据增强 | 低 | 防过拟合 |
| P2 | KV 压缩移植 | 中 | 推理加速 1.5x |
| P3 | RL w/ CER | 高 | 直接优化指标 |
| P4 | 无损量化 | 中 | 模型体积 ↓50% |
