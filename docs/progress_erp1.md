# ERP1 训练进度记录

## 概览

| 项目 | 内容 |
|------|------|
| 阶段 | erp v2 阶段 1 |
| 目标 | 让 base whisper-small 快速适应粤语生活场景数据 |
| 结果 | CER = **9.96%**（best） |
| 状态 | ✅ 完成 |

## 配置

| 参数 | 值 |
|------|-----|
| 基座 | `openai/whisper-small` |
| Attention | `sdpa`（erp1 不开 τ-opus） |
| 训练范围 | decoder-only，encoder 冻结 |
| Epoch | 2 |
| Effective batch | 16（BATCH=4, GRAD_ACCUM=4） |
| LR | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts(T_0=200, T_mult=2) |
| Precision | bf16 |
| EMA | 启用，衰减 0.999 |
| SparseGrad | k_ratio=0.3，固定 |
| τ-opus | 关闭 |
| 优化器 | AdamW |

## 训练日志

### Epoch 1
- 耗时：~7 min
- val_CER = 10.62%
- loss 从 ~0.4 降至 ~0.3

### Epoch 2
- val_CER = **9.96%**（best）
- 产物：`/hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt`（462 MB）

## 产物

- `/hy-tmp/dimsum/outputs/base-erp1/best_model/model.pt` — 最佳 checkpoint（CER 9.96%）
- `/hy-tmp/dimsum/outputs/base-erp1/final_model/model.pt` — 最终 checkpoint
- 训练日志：`/hy-tmp/dimsum/erp1.log`

## 后续

ERP1 权重用作：
1. **ERP2 的 τ* 估计基座**
2. **ERP3 的全参微调初始化**
