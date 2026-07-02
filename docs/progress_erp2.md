# ERP2 τ* 估计进度记录

## 概览

| 项目 | 内容 |
|------|------|
| 阶段 | erp v2 阶段 2 |
| 目标 | 基于 ERP1 权重，估计每个 attention head 的最优 τ* |
| 结果 | 36 modules, 432 heads 的 τ* 分布 |
| 状态 | ✅ 完成 |

## 配置

| 参数 | 值 |
|------|-----|
| σ 函数 | encoder_self: `softplus`；decoder_self: `sigmoid`；decoder_cross: `exp` |
| 估计方法 | Newton 迭代 30 步 |
| 采样策略 | encoder: 8 query × 256 key；decoder_self: 全序列；decoder_cross: 256 key |
| 运行次数 | 5 次，不同随机子集 |
| 聚合方式 | 每头取中位数 |
| 基座权重 | ERP1 best_model |

## τ* 结果

| attention 类型 | τ* mean | τ* min | τ* max |
|---------------|---------|--------|--------|
| encoder_self | 2.53 | 1.31 | 10.00 |
| decoder_self | 2.94 | 1.89 | 6.98 |
| decoder_cross | 1.08 | 1.00 | 1.45 |

### 关键观察

1. **decoder_cross 的 τ* 接近 1.0**：交叉注意力中 softmax（τ=1）已接近最优，说明 whisper 的 cross-attention 预训练分布已足够聚焦。
2. **encoder_self 分布最广**（1.31–10.00）：不同 head 和层对聚焦度的需求差异大，τ 可学习在此处收益最大。
3. **decoder_self τ* 略高于 encoder**：自回归生成中，decoder self-attention 需要更强的聚焦以维持语言连贯性。
4. **分类型 σ 配置合理**：softplus（encoder_self）允许梯度平滑变化；sigmoid（decoder_self）将输出限制在 (0,1) 以稳定自回归；exp（decoder_cross）确保 τ 接近 1 时 σ 接近恒等映射。

## 产物

- `/hy-tmp/dimsum/outputs/erp2_tau_star.json` — per-head τ* 中位数（36 modules, 432 heads）

## 后续

τ* 用作 ERP3 的 `log_tau` 初始化（per-head），训练中自由优化，不加强制正则。
