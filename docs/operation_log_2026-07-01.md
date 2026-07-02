# 操作记录 — 2026-07-01

## 当前目标
为「点心杯」粤语 ASR 初赛训练并优化基于 whisper-small 的模型，重点解决：
1. 得到可靠的全模型 τ* 分布（encoder/decoder/cross），用于后续 τ-opus 与 KV 压缩。
2. 训练一个「base-iop」快速基准模型，作为 τ* 估计的 better base。
3. 最终训练 E-v2（τ 可学习 + 数据增强 + 循环退火），冲击更低 CER。

## 已完成的操作

### 1. 远程环境连通
- 服务器：`root@i-2.gpushare.com:28614`
- 已验证 Python 3.11、PyTorch、transformers、triton 3.1.0 可用。
- 数据集位置：`/root/dimsum/data/prepared/{train,val}.jsonl`（train 8449 条，val 445 条）。

### 2. τ* 估计器迭代
| 版本 | 改动 | 结果 | 问题 |
|------|------|------|------|
| compute_tau_star_on_b.py | 全模块（encoder+decoder+cross） | OOM | 模块数多、decoder 生成序列长、内存未释放 |
| compute_tau_star_b.py | 仅 encoder self-attn；Newton 迭代；20 样本/run；10 runs | 成功：encoder τ* mean=3.05±1.96 | 只算了 encoder，不完整 |

### 3. base-iop 训练启动
- 脚本：[scripts/train_base_iop.py](file:///F:/τ/点心杯/scripts/train_base_iop.py)
- 配置：2 epochs，batch=2，grad_accum=4，effective batch=8，LR=1e-4，decoder-only。
- 调度：`CosineAnnealingWarmRestarts(T_0=200, T_mult=2)`，实现「前期快周期、后期慢退火」。
- 当前进度（训练进行中）：
  ```text
  Epoch 1/2:  69%|██████▊   | 2903/4225 [06:27<02:43, 8.09it/s, loss=0.4481, lr=9.46e-05, step=720]
  ```
- epoch 1 结果：avg_loss=0.3962，val_CER=588.29%（greedy，需观察 epoch 2 是否大幅下降）。
- 产物目录：`/root/dimsum/outputs/base_iop/`

### 4. τ* 求解器组件化（已完成）
- 独立算子：[scripts/tau_star_solver.py](file:///F:/τ/点心杯/scripts/tau_star_solver.py)
  - 提供 `solve_tau_star()`（batched Newton）和 `solve_tau_star_analytic()`。
  - 可配置 sigma：softplus / sigmoid / exp / relu。
  - 提供 `TauStarEstimator` 收集 score 并聚合输出 JSON。
- 全模型 τ* 估计脚本：[scripts/compute_tau_star_iop.py](file:///F:/τ/点心杯/scripts/compute_tau_star_iop.py)
  - 基于 base-iop 权重，采集 encoder/decoder/cross 分数。
  - 采样策略：encoder (8 query × 256 key)、decoder self 全保留、decoder cross 256 key。
- 结果：[outputs/base_iop_tau_star.json](file:///F:/τ/点心杯/outputs/base_iop_tau_star.json)
  - encoder self: τ mean=2.53 (1.31–10.00)
  - decoder self: τ mean=2.59 (1.53–7.07)
  - decoder cross: τ mean=1.12 (1.00–1.44)

### 5. 泛化能力提升计划
见下方「关于泛化」。

## 待解决的关键问题：τ* 估计器开销

### 为什么 full-model τ* 仍然很重？
1. **decoder 是序列生成**：每个样本要跑 `generate()`，每步都要前向，decoder self-attn 和 cross-attn 的 score 都是逐步产生的，不能一次前向拿到。
2. **模块数翻倍**：encoder-only 144 头；加上 decoder self + cross 后变成 432 头，且 decoder 步骤还要乘以输出 token 数（~30–80）。
3. **存储与搬运**：每步 score 要么立刻处理、要么暂存；Python 逐 sample/逐 head 求解会引入大量 CPU-GPU 同步和内存碎片。
4. **生成用 eager attention**：为了 hook 分数，必须关掉 SDPA/flash-attn，生成速度本身会下降。

### 减少开销的思路
1. **采样而非全量**：只对少量短样本（20 条以内）、短输出（max_new_tokens≤32）做生成。
2. **只采关键位置**：decoder self-attn 只保留**当前生成 token** 的 score；cross-attn 对 3000 帧 key 随机采样到 256 帧。
3. **聚合后批量求解**：把同一 module 的所有样本/所有 head/所有 step 的 score 拼成一个大 tensor `(N_total, T_key)`，调用一次 batched Newton，避免 Python 循环。
4. **算子级优化**：把 Newton 求解写成独立可复用的 `tau_star_solver.py` 组件，对外只暴露 `solve_tau_star()` 与 `TauStarEstimator`；必要时用解析近似 `τ ≈ Cov(s, log σ) / Var(log σ)` 做快速预估计。
5. **分阶段估计**：先用 base-iop 跑一次 full-model τ*，后续 E-v2 训练期间固定使用，不再重复估计。

## 竞争力评估（季军可能性）
- 当前 B 模型 val CER ≈ 8.09%。
- 如果榜单头部已经出现接近 100% 句子准确率 / 很低 CER 的结果，季军门槛可能落在 CER 5% 以下。
- 可拉升精度的杠杆：
  - E-v2：τ 可学习 + 循环退火 + 时序反转/MixSpeech 数据增强；
  - 伪标签自训练（用 base-iop 在高置信度无标签/半标签数据上扩展训练集）；
  - beam search + 语言模型重打分；
  - 模型融合 / checkpoint 平均；
  - 文本后处理（粤语文本规范化、同音字纠错）。
- 现在判断能否进季军还为时尚早，需要先跑出 E-v2 的 val CER，再与已知成绩对比。

## 关于泛化：怎么让模型不只记住训练集

1. **数据层面**
   - **时序反转**：已有，让 encoder 不依赖时间顺序 shortcut。
   - **MixSpeech / Mixup**：小比例混合两条音频/标签，增加样本多样性。
   - **SpecAugment**：在 mel 频谱上做时域/频域 mask，模拟噪声和遮挡。
   - **速度扰动**：±10% 变速不变调，增强对语速的鲁棒性。
   - **加性噪声**：混入真实场景噪声（餐厅、街道）。
   - **伪标签自训练**：用 base-iop/E-v1 在高置信度无标签数据（Common Voice yue、复赛部分音频）上生成伪标签，扩充训练集。

2. **模型/优化层面**
   - **τ 可学习 + τ* 初始化**：每个 head 自己找最优聚焦度，避免固定 τ 导致的分布失配。
   - **输入自适应 τ**：用轻量 router 根据音频内容动态调 τ，对清晰语音锐化、对噪声语音平滑。
   - **SparseGrad 课程**：先宽后窄的 top-k 梯度更新，隐式正则化。
   - **权重平均（SWA/EMA）**：对多个 checkpoint 做指数移动平均，得到更平滑、泛化更好的权重。
   - **Dropout / weight decay**：已设 0.01，可适当加大。

3. **验证层面**
   - 用 **Common Voice Cantonese (yue)** 或官方 test 集作为 OOD 验证，不仅看 val CER。
   - 每轮训练同时记录 **in-domain CER** 和 **OOD CER**，防止过拟合训练集。

4. **推理层面**
   - **Beam search + 语言模型重打分**：降低局部解码错误。
   - **Checkpoint 平均 + 多模型投票**：ensemble 能显著压低 CER。
   - **文本后处理**：粤语常用字规范化、同音字纠错。

## 规划更新：三阶段 erp v2（2026-07-01 晚）

旧服务器已弃用，新 GPU 机器正在准备。训练策略重新梳理为三阶段，详见 [docs/training_plan_erp_v2.md](file:///F:/τ/点心杯/docs/training_plan_erp_v2.md)。

| 阶段 | 产物 | 关键决策 |
|------|------|---------|
| erp1 | `base-erp1` | 重训；decoder-only；bf16；SparseGrad k=0.3；τ-opus **关闭**；EMA |
| erp2 | `erp2_tau_star.json` | 基于 erp1 估计 per-head τ*；5~10 runs 取中位数 |
| erp3/E-v2 | `best_model` | τ 可学习，用 erp2 的 τ* 初始化；训练中**不加强制正则**；SparseGrad 课程；时序反转 + 硬标签 MixSpeech；bf16 |

核心变更：
- **全链路 bf16**。
- **τ 用 erp2 的 τ* 初始化**，但训练中自由优化（不加 `(τ-τ*)²` 正则）。
- **erp1 重训**，因为新 GPU 环境需要重新配置，且旧 base-iop 产物在旧服务器上。
- MixSpeech 先用**硬标签简化版**，时间充裕再上软标签。

## 6. 新 GPU 机器（L40S-48GB）环境就绪
- SSH：`ssh -p 59010 root@i-1.gpushare.com`，密码 `HMAV6TEcCARYegYCFEGB6B89sDqqSGfU`。
- GPU：NVIDIA L40S，46068 MiB，driver 535.230.02，CUDA 12.2。
- PyTorch：2.4.0+cu121，`torch.cuda.is_available() == True`。
- Workspace：`/hy-tmp/dimsum/{scripts,data,outputs}`，`/hy-tmp/hf_cache`。
- 磁盘：`/hy-tmp` 50G 可用，`/` 29G 可用。

## 7. τ-opus 训练算子落地并验证
- 上传 [scripts/stau_opus_operator.py](file:///F:/τ/点心杯/scripts/stau_opus_operator.py) 与 [scripts/_test_stau_opus_grad.py](file:///F:/τ/点心杯/scripts/_test_stau_opus_grad.py) 到 `/hy-tmp/dimsum/scripts/`。
- 远程运行 `python3 _test_stau_opus_grad.py` → **gradcheck: True**，前向/反向数值稳定。
- [scripts/patch_whisper_stau.py](file:///F:/τ/点心杯/scripts/patch_whisper_stau.py) 已一并上传，用于把算子接入 Whisper `eager_attention_forward`。
- 注意：`apply_learnable_stau` 的 `default_tau_json` 默认仍为 Windows 路径，调用时需显式传入 `/hy-tmp/dimsum/outputs/...` 路径，或在启动脚本中统一覆盖。

## 8. 数据与路径问题
- `/hy-tmp/dimsum/data` 目前为空；需要重新下载 `leeduckgo/cantonese-life-scenarios-corpus` 并 prepare 成 `train.jsonl` / `val.jsonl`。
- 当前训练脚本硬编码 `/root/dimsum/...`，需适配到 `/hy-tmp/dimsum/...` 或通过环境变量注入。
- **已启动数据下载**：`prepare_data_remote.py` 正在后台运行（PID 6538），从 HF 拉取 `data.zip` / `index.csv` / `test.jsonl` / `evaluator.py` 并解压/映射。日志：`/hy-tmp/dimsum/prepare_data.log`。
- **data.zip 大小**：约 1.6 GB；当前已下载约 50 MB，按当前速度预计还需 2–3 小时。
- **已部署自动启动脚本**：`wait_and_start_erp1.sh`（PID 6893）会在 `train.jsonl`/`val.jsonl` 就绪后自动启动 erp1。日志：`/hy-tmp/dimsum/wait_and_start_erp1.log`，训练日志将写入 `/hy-tmp/dimsum/erp1.log`。
- **erp1 速度预估**：batch=4、grad_accum=4，每 epoch 约 2112 个 micro-batch；参考旧机 8 it/s，L40S+bf16 估计 8–12 it/s，单 epoch 训练约 3–5 分钟，greedy 评估约 1–2 分钟，2 epochs 合计约 10–15 分钟。

## 下一步计划
1. 在新机器下载并 prepare 初赛数据集到 `/hy-tmp/dimsum/data/prepared/`。
2. 更新训练脚本路径/env 变量，使其在新机器可跑。
3. 启动 erp1（base-erp1）训练：decoder-only、bf16、SparseGrad k=0.3、τ-opus 关闭。
4. erp1 完成后跑 erp2（τ* 估计，5~10 runs 取中位数）。
5. erp3/E-v2：τ 可学习（erp2 τ* 初始化）+ 数据增强 + SparseGrad 课程。
6. 产物回传本地，做 ablation 与提交。

---
记录时间：2026-07-01
