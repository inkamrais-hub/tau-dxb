# Research Note 2026-07-01 — Base-XL 初赛训练与评测

## 实验 ID
EXP-001: Base-XL (τ-opus + SparseGrad-WTA + decoder 全量微调)

## 模型架构
- Base: `openai/whisper-small` (244M 参数)
- Trainable: decoder only (153.6M / 241.7M = 63.5%)
- τ-opus: 36 attention 模块替换 (encoder_self=softplus, decoder_self=sigmoid, decoder_cross=exp, per-head τ*)
- SparseGrad-WTA: 24 层 FFN 梯度稀疏 (k=30%)
- Encoder 冻结

## 训练配置
- 数据: 8449 训练 / 445 验证 (来自初赛语料集)
- Batch: 4, Grad Accum: 4, 有效 batch=16
- 学习率: 1e-5, CosineAnnealing, weight_decay=0.01
- Epochs: 5 (总约 75 分钟)
- 设备: RTX 3080 20GB

## 训练 Loss 曲线
| Epoch | Avg Loss | 趋势 |
|-------|----------|------|
| 1     | 0.4136   | 模型适应 τ-opus 分布 |
| 2     | 0.1738   | ↓ 大幅下降 |
| 3     | 0.0969   | ↓↓ |
| 4     | 0.0566   | ↓↓↓ |
| 5     | 0.0376   | ↓↓↓↓ 收敛 |

## 评测结果 (官方 evaluator)
- **CER: 0.54%** (0.005449)
- **字符准确率: 99.46%**
- **句子准确率: 95.63%** (1817/1900 句完全正确)
- 语气词召回 (㗎/喇/啫): 完全正确

### 典型错误
1. 同音字混淆: "美术→理术", "古诗→古识", "课间→坐间"
2. 缺字: "要先剪→先剪"
3. 复杂错误: "组装器材→组终起才"

## 知识状态更新
- [FACT ★] SparseGrad-WTA + τ-opus 在 whisper-small 上前向+反向均稳定
- [FACT ★] τ-opus 的正确闭式反传（τ·σ^(τ-1)·σ' + argmax 修正）数值稳定
- [FACT ★] Base-XL 在当前测试集上达到 CER 0.54%
- [HYPOTHESIS] τ-opus 的 softplus/sigmoid σ 函数提供了 softmax 无法表达的注意力形状

## 下一步计划 (待用户确认)
1. 回传产物到本地
2. 运行 zero-shot baseline 做对比
3. 设计迭代训练策略 (τ 课程学习、MixSpeech、RL)
4. τ*-KV 压缩移植到 whisper
