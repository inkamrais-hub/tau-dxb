# 点心杯 ASR 三阶段训练规划 — erp v2

## 一、总览

| 阶段 | 名称 | 目标 | 主要产物 | 状态 |
|------|------|------|---------|------|
| 1 | erp1 | 让 base 模型快速适应粤语生活场景数据 | `base-erp1` 权重 | ✅ CER=9.96% |
| 2 | erp2 | 基于 erp1 估计 per-head τ* 分布（分类型 σ） | `erp2_tau_star.json` | ✅ 完成 |
| 3 | erp3 | 全参微调 + τ-opus + WTA前向稀疏 + 剪枝 + SparseGrad | `best_model` | 脚本就绪，待训练 |

**核心原则**：
- 全链路 **bf16** 训练/推理。
- τ* 仅用于**初始化**可学习 τ，训练中自由优化，不加强制正则。
- τ 全程可学习（encoder self / decoder self / decoder cross），不固定。
- σ 函数按 attention 类型分别配置（和 erp2 一致）。

---

## 二、阶段 1：erp1 — Base 适应训练 ✅

### 配置

| 参数 | 值 |
|------|-----|
| 基座 | `openai/whisper-small` |
| attention | `sdpa`（erp1 不开 τ-opus） |
| 训练范围 | decoder-only，encoder 冻结 |
| Epoch | 2 |
| Effective batch | 16（BATCH=4, GRAD_ACCUM=4） |
| LR | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts(T_0=200, T_mult=2) |
| Precision | **bf16** |
| EMA | 启用，衰减 0.999 |
| SparseGrad | k_ratio=0.3，固定 |
| τ-opus | **关闭** |

### 结果

- Epoch 1: val_CER=10.62%
- Epoch 2: val_CER=**9.96%**（best）
- 产物：`/hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt`（462 MB）

---

## 三、阶段 2：erp2 — τ* 估计 ✅

### 配置

| 参数 | 值 |
|------|-----|
| σ 函数 | encoder self: `softplus`；decoder self: `sigmoid`；decoder cross: `exp` |
| 采样 | encoder: 8 query × 256 key；decoder self: 全序列；decoder cross: 256 key |
| 迭代 | Newton 30 步 |
| 运行次数 | 5 次，不同随机子集 |
| 聚合 | 每头取中位数 |

### 结果

| attention 类型 | τ* mean | τ* min | τ* max |
|---------------|---------|--------|--------|
| encoder self | 2.53 | 1.31 | 10.00 |
| decoder self | 2.94 | 1.89 | 6.98 |
| decoder cross | 1.08 | 1.00 | 1.45 |

- 产物：`/hy-tmp/dimsum/outputs/erp2_tau_star.json`（36 modules, 432 heads）

---

## 四、阶段 3：erp3 — 最终训练

### 配置

| 参数 | 值 |
|------|-----|
| 训练范围 | **全参微调**（encoder + decoder） |
| Epoch | 5 |
| Effective batch | 16 |
| LR (base) | 3e-5 |
| LR (τ/α) | 1e-3 |
| Scheduler | 三阶段 LambdaLR：epoch 1-2 快周期退火 / epoch 3-4 慢退火 / epoch 5 warm restart → 0 |
| Precision | **bf16** |
| Gradient clipping | 1.0 |
| EMA | 启用，衰减 0.999 |

### 组件分配

| 组件 | encoder | decoder |
|------|---------|---------|
| FFN 前向 | WTA 激活稀疏（top-k） | 全量 |
| FFN 反向 | WTA 梯度稀疏（同一 mask） | SparseGrad 梯度稀疏 |
| 剪枝 | 有（第3epoch起） | 无 |
| attention | τ-opus 可学习（softplus） | τ-opus 可学习（self=sigmoid, cross=exp） |

### τ-opus 可学习

- 每个 WhisperAttention 挂 `STauOpusLearnable`（融合算子，自定义 backward）。
- `log_tau` 初始化为 erp2 的 τ* 值（per-head）。
- `log_alpha` 初始化为 0（α=1.0）。
- **训练中完全自由优化**，不加 `(τ - τ*)²` 正则。
- σ 分类型：encoder_self=softplus, decoder_self=sigmoid, decoder_cross=exp。

### WTA 前向稀疏 + SparseGrad 课程

| Epoch | encoder k_ratio | decoder k_ratio |
|-------|----------------|----------------|
| 1 | 0.60 | 0.50 |
| 2 | 0.55 | 0.45 |
| 3 | 0.50 | 0.40 |
| 4 | 0.45 | 0.35 |
| 5 | 0.40 | 0.30 |

- encoder: `WTA_Activation_Function`（前向 top-k 稀疏 + 反向同一 mask）
- decoder: `SparseGradWTA_Function`（前向恒等，反向 top-k 梯度）

### 训练时软剪枝（动态可复活）

| 阶段 | 操作 |
|------|------|
| Epoch 1-2 | 只统计 win count，不剪枝 |
| Epoch 3 起 | 每 epoch 初：unfreeze_all → prune_low_winners → reset_win_counts |

- `win_rate = win_count / mean(win_count)`，均匀分布时 = 1.0
- **软剪枝**：pruned 神经元的**梯度冻结**（通过 `register_hook` mask 梯度），**权重保持原值**
- **前向不封死**：pruned 神经元仍参与 WTA top-k 选择，若被频繁选中 → win rate 上升 → 下 epoch `unfreeze_all` 后不再被剪 → **自然复活**
- 梯度 hook 在 `apply_sparse_grad_wta` 时注册一次，hook 内部从全局 `_GRAD_MASKS` 读取，清空后自动失效

### 数据增强

| 增强 | 概率 | 说明 |
|------|------|------|
| 时序反转 | 0.2 | 只反转音频，文本不变 |
| MixSpeech（硬标签） | 0.1 | 音频按 9:1 混合，标签用主样本 |

### 产物

```text
/hy-tmp/dimsum/outputs/erp3/
├── best_model/model.pt
├── final_model/model.pt
├── epoch{1..5}/model.pt
├── tau_epoch{1..5}.json
└── log.txt
```

---

## 五、远程路径

| 资源 | 路径 |
|------|------|
| workspace | `/hy-tmp/dimsum/` |
| 训练数据 | `/hy-tmp/dimsum/data/prepared/{train,val}.jsonl` |
| 模型缓存 | `/hy-tmp/whisper-small-local/` |
| HF cache | `/hy-tmp/hf_cache/` |
| erp1 产物 | `/hy-tmp/dimsum/outputs/base-erp1/` |
| erp2 产物 | `/hy-tmp/dimsum/outputs/erp2_tau_star.json` |
| erp3 产物 | `/hy-tmp/dimsum/outputs/erp3/` |

---

## 六、消融实验

| 实验 | 变量 | 目的 |
|------|------|------|
| A | erp3 去掉 τ-opus | 验证 τ-opus 收益 |
| B | erp3 去掉 SparseGrad+WTA | 验证稀疏收益 |
| C | erp3 去掉 MixSpeech+时序反转 | 验证数据增强收益 |
| D | erp3 去掉剪枝 | 验证剪枝收益 |
| E | erp3 τ 初始 1.0 vs τ* | 验证 erp2 是否真有用 |

时间紧时至少做 A、B、E 三项。每项从 erp1 best_model 开始训 3-5 epoch。

---

## 七、KV 压缩（推理优化，训练后）

训练完成后，用 τ* 指导 head 级 KV 剪枝：
- 高 τ* head：注意力聚焦，可以激进剪 KV（保留少量关键帧）
- 低 τ* head：注意力分散，保留更多 KV 帧
- 最小保留帧数 K_MIN=64

这是推理加速，不影响训练。

---

## 八、算子实现状态

| 算子 | 文件 | 状态 |
|------|------|------|
| τ-opus 融合算子 | `stau_opus_operator.py` | ✅ 自定义 backward，bf16 兼容 |
| SparseGrad | `patch_whisper_sparse_grad.py` | ✅ 前向恒等，反向 top-k |
| WTA 激活 | `patch_whisper_sparse_grad.py` | ✅ 前向+反向 top-k |
| WinCountTracker | `patch_whisper_sparse_grad.py` | ✅ 统计 + 剪枝 |
| τ-opus patch | `patch_whisper_stau.py` | ✅ 分类型 σ，可学习 τ/α |

所有算子已通过冒烟测试（bf16 前向+反向，无 NaN/Inf）。
