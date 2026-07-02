# tau-dxb — 点心杯粤语 ASR (Echo-Opus 路线)

## 路线说明
- **erp 路线已废弃** (erp1/erp2/erp3/erp4)
- **当前路线: Echo-Opus-1** — τ-opus 注意力归一化 + WTA 稀疏激活 + 全参微调

## 目录结构
```
echo/           # Echo-Opus 训练核心
  train_echo_opus1.py      训练主脚本
  launch_echo_opus1.sh     启动脚本
  patch_whisper_stau.py    τ-opus 注意力 patch
  stau_opus_operator.py    τ-opus eager 算子
  stau_opus_flash_v2.py    τ-opus Flash V2 Triton 算子
  patch_whisper_sparse_grad.py  WTA 稀疏 + 软剪枝
  triton_wta.py            WTA Triton kernel
  predict.py               推理脚本
  remote_git_push.py       每 epoch 自动上传
tau_star/       # τ* 求解器
  compute_tau_star_echo.py τ* 估计器
  tau_star_echo_opus1.json τ* 求解结果 (36 层 × 12 head)
infra/          # 基础设施
  remote.py                paramiko 远程管理
  monitor_local.py         本地长期监控
```

## τ-opus 核心公式
attn = σ(s)^τ / Σσ(s)^τ

σ 分类型:
- encoder_self: softplus
- decoder_self: sigmoid
- decoder_cross: exp (τ=1 时精确等价 softmax)

τ 初始化: 从 τ* 估计器加载 (KL 散度最小化解)
τ 参数化: τ = softplus(log_tau) + 1.0 > 1 (可学习)

## WTA 稀疏
- encoder FFN: wta_activation (前向 top-k 稀疏 + 反向 top-k 梯度)
- decoder FFN: sparse_grad (前向恒等, 反向 top-k 梯度)
- k_ratio: 可学习 (LearnableKRatio, sigmoid 参数化)
- 软剪枝: prune_low_winners() 已实现 (win rate < 5% 冻结梯度)

## 基础模型
openai/whisper-small (241M params, 全参微调)
