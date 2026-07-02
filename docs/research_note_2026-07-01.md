# 点心杯 ASR 研究笔记 — 2026-07-01

## 1. 赛题约束（来自 PDF 解读）

- 基座模型必须是 `openai/whisper-small`。
- 微调方式不限：Full / LoRA / Prompt tuning 均可。
- 评测指标：**CER（字错误率）**，越低越好。
- 输出必须是粤语文本（汉字），不能是粤拼。
- 音频预处理：16 kHz、单声道。
- 文本规范化：全角半角、标点、繁简体。
- 允许数据增强：SpecAugment、速度扰动等。
- **禁止**：人工修改测试结果、调用在线 API 作弊。

> 注意：赛事说明第九节末尾仍有"必须使用提供的大模型接口、不可训练独立模型、不可引入外部数据"的冲突语句。建议再向主办方确认是否适用于决赛/复赛，但 PDF 中初赛规则明确允许 whisper-small 微调。

## 2. 检索到的外部方法

### 2.1 粤语 ASR 直接相关

1. **CantoASR (2025-11)**
   - 方法：强制对齐提取 F0/斜率/时长等韵律特征 → LoRA 微调 Whisper-Large-V3 → 用 Qwen2-Audio 做韵律感知纠错。
   - 启示：声调/韵律信息对粤语识别至关重要；可以用 LoRA 把韵律特征注入注意力或解码器。
   - 来源：https://arxiv.org/pdf/2511.04139v1

2. **LoRA-INT8 Whisper (2025-09)**
   - 方法：Whisper-tiny + LoRA rank=8 在 Common Voice zh-HK 微调，CER 从 49.5% 降到 11.1%，接近全量微调（10.3%）。
   - 启示： whisper-small 用 LoRA 微调即可达到接近全量效果，且省显存。
   - 来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC12431075/

3. **whisper-small-cantonese（HuggingFace）**
   - 已微调好的 whisper-small，在 Common Voice 16.0 上 CER 7.93%（无标点）。训练数据 934h，含 Common Voice、CantoMap、YouTube 伪标注。
   - 启示：粤语 whisper-small 微调天花板已经较低；我们重点是赛道给定的 2GB 生活场景数据 + 复赛藤县方言。
   - 来源：https://www.promptlayer.com/models/whisper-small-cantonese

4. **Phoneme-Aware Hierarchical Augmentation (2025-07)**
   - 方法：基于强制对齐的 Phoneme Substitution Matrix（PSM）+ 语义感知 SpecAugment（wav2vec attention heatmap + RL 控制掩码区域）。
   - 效果：Common Voice 粤语 50h，wav2vec CER 26.17% → 16.88%；Zipformer 38.83% → 23.55%。
   - 启示：音素/粤拼层面的对抗替换 + 自适应时频掩码是有效低资源增强。
   - 来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC12298586/

5. **WenetSpeech-Yue (2025-09)**
   - 方法：2.18 万小时大规模粤语语料，多维度标注；用 SenseVoice / Whisper / 商业模型做伪标注。
   - 启示：大规模伪标注能显著提升粤语 ASR，但初赛/复赛可能受限于"不可引入外部数据"。
   - 来源：https://arxiv.org/html/2509.03959v2

6. **FSR-2025 台湾客语 baseline（ROCLING 2025）**
   - 方法：Whisper-large-v2/v3 + LoRA(rank=16, α=32, dropout=0.05)，label smoothing，attention+MLP LoRA，beam=5，temperature=0.0，无外部数据/LM。
   - 启示：低资源方言 ASR 的可靠配方；可直接迁移到粤语微调。
   - 来源：https://preview.aclanthology.org/manual-author-scripts/2025.rocling-main.55.pdf

## 3. 现有代码资产盘点

### 3.1 τ-opus 注意力归一化

| 文件 | 作用 |
|------|------|
| `f:\τ\τopus\stau_opus.py` | τ-opus 核心：`attn = σ(s)^τ / Σσ(s)^τ`，含可学习 τ/α、σ 混合、自定义 autograd Function、GPT-2 monkey-patch。 |
| `f:\τ\τopus\tau_star_opus.py` | τ* 估计器：`τ* = Cov(s, log σ(s)) / Var(log σ(s))`，支持多种 σ 函数与逐层回归系数。 |
| `f:\τ\τ-softplus\attention_mechanisms\s_tau_fused_attention_v15.py` | CUDA kernel 版 τ-opus attention，fp16、分块、向量加载，适用于高效训练。 |
| `f:\τ\τopus\run_gpt2_opus_mix.py` | 用可学习 σ 混合（softplus+sigmoid）在 GPT-2 上训练 α/β，τ 初始化为 τ* 并冻结。 |

核心可复用组件：
- `STauOpusMaxStableFn`：带 max-stable 的 σ^τ/Σσ^τ 注意力。
- `STauOpusLearnable`：每头可学习 τ、α。
- `LearnableSigmaMixture`：每头可学习 σ 混合权重。
- `TauStarOpus.softplus_closed()`：从当前 attention scores 在线估计 τ*。

### 3.2 表征控制工具箱（repeng_toolkit）

| 文件 | 作用 |
|------|------|
| `f:\τ\repeng\repeng_toolkit\steering.py` | TokenBooster、LogitController、NormConstrainedController、ActivationSteerer、HiddenStateGuider。 |
| `f:\τ\repeng\repeng_toolkit\drm.py` | DRM / DualProbe / FlowDecomposer：通过扰动对偶关系定位信息流破裂点并做层间控制。 |
| `f:\τ\repeng\experiments\_qwen_xxt_representation_steering.py` | XXT-style：对 hidden state 协方差做 PCA/SVD，提取自然控制轴，再在指定层加减方向向量。 |

核心可复用组件：
- `ActivationSteerer`：在隐藏层注入固定方向向量。
- `HiddenStateGuider`：从 correct/wrong pairs 学习引导向量。
- `TokenBooster`：在解码 logit 上直接增强/抑制特定 token（如粤语语气词、高频错字）。
- `FlowDecomposer`：把 hidden state 分解为输入流/输出流/内部流，定位最佳干预层。

## 4. 融合方案：τ-opus + 表征控制 → Whisper-small

### 4.1 核心创新点

把两套自研代码迁移到语音领域，解决粤语 ASR 的两个痛点：
1. **声调/音素区分难** → τ-opus 注意力替代 softmax，让每头可学习"尖锐/平坦"程度，可能更好保留细微声学差异。
2. **粤语风格/语气词/方言词易错** → 表征控制在 encoder/decoder 隐藏层注入"粤语方向"向量，并在输出 logit 增强粤语特色 token。

### 4.2 τ-opus 接入 Whisper-small

实现思路（不改动预训练权重，只 monkey-patch 注意力）：
1. 在 `WhisperAttention` 的 `forward` 中，计算 QK scores 后，把 `F.softmax` 替换为 `STauOpusMaxStableFn.apply`。
2. 提供两种模式：
   - **固定 τ* 模式**：用 `TauStarOpus.softplus_closed()` 对每层每头在线估计 τ*，冻结。
   - **可学习模式**：用 `STauOpusLearnable(n_heads)` 把 τ、α 作为可训练参数。
3. 处理 mask：Whisper encoder 是双向，decoder 是因果+cross-attention。τ-opus 函数前先对 masked 位置加 `-1e9`。
4. 稳定性：先跑小规模（如 100 batch）冒烟测试，确认 fp16/bf16 不 nan；必要时在 fp32 中计算 σ^τ 再回传。

风险：替换注意力会改变梯度流，可能导致训练不稳定；需要与 LoRA 联合调参。

### 4.3 表征控制增强

实现思路：
1. **语言/风格方向向量**：
   - 用训练集中粤语文本 vs 普通话译文分别过微调后的 encoder，取某层 hidden state 均值差作为"粤语方向"。
   - 在 encoder 深层（如 L8-L10）加上 `+ λ * cantonese_axis`，强化粤语声学表征。
2. **语气词增强（TokenBooster）**：
   - 从训练集统计高频语气词（㗎、喇、啫、嘅、喎、啰 等）。
   - 解码时给这些 token 的 logit 加 small bias（如 +0.5），提升语气词召回率（评测指标之一）。
3. **错误纠正方向（HiddenStateGuider）**：
   - 构造正确/错误 pairs：例如把 Whisper zero-shot 预测结果作为"错误"，ground truth 作为"正确"。
   - 在中间层学习一个引导向量，让 hidden state 朝正确方向偏移。

### 4.4 初赛完整 pipeline

1. **数据预处理**
   - 16k 单声道、VAD 切边、响度归一化。
   - 文本清洗：NFKC、全角半角、繁简转换、英文小写、粤语异体字映射。
2. **基线**
   - whisper-small zero-shot CER。
3. **LoRA 微调基线**
   - LoRA on `q_proj, k_proj, v_proj, out_proj, fc1, fc2`，rank=16，α=32，dropout=0.05，label smoothing=0.1。
   - Beam=5，temperature=0.0，early stopping on dev CER。
4. **τ-opus 实验组**
   - 在上述 LoRA 基线基础上，把 attention softmax 替换为 τ-opus（可学习 τ）。
   - 对比 CER、训练稳定性、收敛速度。
5. **表征控制实验组**
   - 训练后用 XXT/FlowDecomposer 提取粤语方向向量，在 encoder 深层做 inference-time steering。
   - 用 TokenBooster 增强语气词。
6. **数据增强**
   - SpecAugment + 速度扰动 0.9-1.1 + 加噪（MUSAN/真实噪声）。
   - 可选：基于粤拼的音素替换（参考 PSM 论文）。

### 4.5 复赛（藤县勾漏片）迁移

1. **热启动**：用初赛最佳 whisper-small 权重。
2. **方言 Adapter**：在 encoder 每层后插入 bottleneck adapter（256 dim），仅训练 adapter + decoder LoRA。
3. **多任务**：把 IPA、汉字、普通话译文作为三个任务 token（`<zh>`, `<ipa>`, `<cmn>`），共用 decoder。
4. **τ-opus 迁移**：把初赛学到的 τ/α 作为初始化，继续在藤县数据上微调。
5. **表征控制**：针对藤县 vs 标准粤语分别提取方向向量，做方言风格引导。

## 5. 风险与待验证假设

| 假设 | 验证方式 |
|------|---------|
| τ-opus 替换 Whisper attention 能稳定训练 | 100-batch 冒烟 + 单 epoch CER 对比 |
| τ-opus 在 ASR 上优于 softmax | 同一 LoRA 配置下 A/B 测试 |
| 表征控制方向向量能改善粤语/语气词输出 | 在 dev 集上对比 steering 前后 CER 与语气词 recall |
| 藤县 IPA 标签格式一致 | 数据下载后统计 token 种类与对齐长度 |
| 外部文本语料/伪标注是否违规 | 邮件主办方确认 |

## 6. 下一步

1. 下载初赛数据集 + evaluator.py，跑 zero-shot CER 基线。
2. 编写 `patch_whisper_stau.py`，把 τ-opus 接入 `WhisperAttention`。
3. 编写 LoRA 训练脚本，先做标准 softmax 基线。
4. 在基线上叠加 τ-opus + 表征控制，做对照实验。
5. 确认复赛规则边界（外部数据、伪标注、模型大小）。
