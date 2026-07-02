#!/usr/bin/env python3
"""
点心杯初赛推理脚本 —— τ-dux

基于 openai/whisper-small 加载微调权重，对测试音频进行粤语 ASR 推理。
输出 JSONL 格式（每行 {"audio_path": "...", "pred_text": "..."}），
兼容官方 evaluator.py 评估。

用法:
    # 基础推理（erp1 权重，标准 whisper）
    python predict.py --input_dir ./test_audio --model model.pt --output result.jsonl

    # τ-opus 推理（erp3 权重，需指定 tau_star 初始化文件）
    python predict.py --input_dir ./test_audio --model model.pt \
        --tau_opus --tau_init tau_star.json --output result.jsonl

修改说明:
    - 本脚本基于官方 Whisper generation API 封装
    - ==新增==: τ-opus 可学习注意力归一化支持（--tau_opus 开关）
    - ==新增==: WTA 稀疏激活 + SparseGrad 梯度稀疏 patch
    - ==新增==: 动态剪枝缓存清除（evaluate 前确保前向完整）
    - ==新增==: bf16 混合精度推理（CUDA 下自动启用）
    - ==未修改==: generate() 核心逻辑、tokenizer、feature_extractor 均为官方 API
"""
import argparse, json, os, sys, glob, time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path):
    """加载音频并统一为 16kHz mono float32"""
    audio, sr = sf.read(path)
    if sr != SAMPLING_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    return audio.astype("float32")


def apply_tau_opus_patches(model, tau_init_path):
    """
    ==新增== 应用 τ-opus 推理阶段 patch:
    1. τ-opus 可学习注意力归一化（STauOpusAttentionFunction）
    2. WTA 稀疏激活 + SparseGrad 梯度稀疏（前向 top-k）
    3. 清除剪枝缓存，保证前向完整

    参考: patch_whisper_stau.py, patch_whisper_sparse_grad.py
    """
    # 动态导入避免未安装 triton 时报错
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
    from patch_whisper_stau import apply_learnable_stau
    from patch_whisper_sparse_grad import apply_sparse_grad_wta, set_sparse_grad_k_ratio

    # Step 1: τ-opus attention
    print("[predict]   applying τ-opus patches...")
    apply_learnable_stau(model, tau_init=tau_init_path)

    # Step 2: WTA + SparseGrad（推理阶段仅前向稀疏生效）
    apply_sparse_grad_wta(model,
                          encoder_k_ratio=0.4,
                          decoder_k_ratio=0.3,
                          track_wins=False)

    # Step 3: 清除剪枝 grad masks，保证前向完整
    from patch_whisper_sparse_grad import clear_all_grad_masks
    clear_all_grad_masks(model)

    print("[predict]   τ-opus patches applied (inference mode)")

    return model


def main():
    parser = argparse.ArgumentParser(description="τ-dux ASR 推理脚本")
    parser.add_argument("--input_dir", default="./test_audio",
                        help="测试音频目录")
    parser.add_argument("--model", default="model.pt",
                        help="微调权重路径")
    parser.add_argument("--output", default="result.jsonl",
                        help="输出 JSONL 路径（每行: {audio_path, pred_text}）")
    parser.add_argument("--base_model", default="openai/whisper-small",
                        help="基座模型（默认: openai/whisper-small）")

    # ==新增== τ-opus 开关
    parser.add_argument("--tau_opus", action="store_true",
                        help="启用 τ-opus 推理（erp3 权重需要）")
    parser.add_argument("--tau_init", default=None,
                        help="τ* 初始化 JSON（erp3 需要，如 tau_star.json）")

    args = parser.parse_args()

    # ── 加载基座模型 ──
    # 官方 API: WhisperForConditionalGeneration.from_pretrained
    print(f"[predict] 基座模型: {args.base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.to(DEVICE)

    # ── 加载微调权重 ──
    if os.path.exists(args.model):
        print(f"[predict] 微调权重: {args.model}")
        # weights_only=True 保证安全加载
        state = torch.load(args.model, map_location="cpu", weights_only=True)
        # strict=False 允许 τ-opus 额外参数（erp3 才有）
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  missing keys (ignored if τ-opus): {len(missing)}")
        if unexpected:
            print(f"  unexpected keys (from checkpoint): {len(unexpected)}")
        print(f"  加载完成, 共 {len(state)} 个参数组")
    else:
        print(f"[predict] 未找到 {args.model}，使用 base 模型直接推理")

    # ── ==新增== τ-opus patches（erp3 权重需要）──
    if args.tau_opus:
        tau_path = args.tau_init or os.path.join(
            os.path.dirname(args.model), "tau_star.json"
        )
        if not os.path.exists(tau_path):
            print(f"[predict] 警告: τ_init 文件 {tau_path} 不存在，使用默认初始化")
            tau_path = None
        apply_tau_opus_patches(model, tau_init_path=tau_path)

    model.eval()

    # ── 加载 processor ──
    # 官方 API: WhisperProcessor
    processor = WhisperProcessor.from_pretrained(args.base_model)

    # ── 推理参数 ──
    # 官方 API: model.generate
    gen_kwargs = {
        "language": "zh",
        "task": "transcribe",
        "num_beams": 1,       # greedy decoding
        "do_sample": False,   # deterministic
    }

    # ── 扫描音频文件 ──
    audio_exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    audio_files = sorted(
        f for f in glob.glob(os.path.join(args.input_dir, "*"))
        if f.lower().endswith(audio_exts)
    )
    if not audio_files:
        # 递归查找
        audio_files = sorted(
            f for ext in audio_exts
            for f in glob.glob(os.path.join(args.input_dir, "**", f"*{ext}"), recursive=True)
        )
    if not audio_files:
        print(f"[predict] 错误: 在 {args.input_dir} 中未找到音频文件")
        sys.exit(1)
    print(f"[predict] 音频文件数: {len(audio_files)}")

    # ── 逐条推理 ──
    # ==新增==: bf16 混合精度推理（CUDA 下自动启用）
    results = []
    for i, audio_path in enumerate(audio_files):
        audio = load_audio(audio_path)

        # 官方 API: WhisperFeatureExtractor
        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features.to(DEVICE)

        # 官方 API: model.generate
        with torch.no_grad():
            if DEVICE == "cuda":
                # ==新增==: bf16 推理加速
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred_ids = model.generate(inputs, **gen_kwargs)
            else:
                pred_ids = model.generate(inputs, **gen_kwargs)

        # 官方 API: tokenizer.batch_decode
        text = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )[0]

        # ==修改==: 输出格式匹配 evaluator.py: {"audio_path", "pred_text"}
        # audio_path 使用 basename 以便与 verify.csv 中的文件名匹配
        results.append({
            "audio_path": os.path.basename(audio_path),
            "pred_text": text,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time if i == 100 else 0
            print(f"  [{i+1}/{len(audio_files)}] {time.strftime('%H:%M:%S')}")

    # ── 写入 JSONL ──
    # ==修改==: 使用 evaluator.py 兼容格式：每行 JSON
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[predict] 完成 → {args.output} ({len(results)} 条)")


if __name__ == "__main__":
    start_time = time.time()
    main()
