#!/usr/bin/env python3
"""
点心杯初赛推理脚本
基于 openai/whisper-small 加载微调权重，对测试音频进行粤语 ASR 推理。

用法:
    python predict.py --input_dir <音频目录> --output result.jsonl
    python predict.py --input_dir <音频目录> --model model.pt --output result.jsonl
"""
import argparse, json, os, sys, glob, time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path):
    audio, sr = sf.read(path)
    if sr != SAMPLING_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    return audio.astype("float32")


def main():
    parser = argparse.ArgumentParser(description="τ-dux ASR 推理")
    parser.add_argument("--input_dir", default="./test_audio",
                        help="测试音频目录")
    parser.add_argument("--model", default="model.pt",
                        help="微调权重路径")
    parser.add_argument("--output", default="result.jsonl",
                        help="输出 JSONL 路径")
    parser.add_argument("--base_model", default="openai/whisper-small",
                        help="基座模型")
    args = parser.parse_args()

    print(f"[predict] 加载基座模型: {args.base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.to(DEVICE)

    if os.path.exists(args.model):
        print(f"[predict] 加载微调权重: {args.model}")
        state = torch.load(args.model, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"  加载完成: {len(state)} 个参数组")
    else:
        print(f"[predict] 未找到微调权重 {args.model}，使用 base 模型")

    model.eval()

    processor = WhisperProcessor.from_pretrained(args.base_model)
    gen_kwargs = {
        "language": "zh",
        "task": "transcribe",
        "num_beams": 1,
        "do_sample": False,
    }

    # 扫描音频文件
    audio_exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    audio_files = sorted(
        f for f in glob.glob(os.path.join(args.input_dir, "*"))
        if f.lower().endswith(audio_exts)
    )
    if not audio_files:
        # 尝试递归查找
        audio_files = sorted(
            f for ext in audio_exts
            for f in glob.glob(os.path.join(args.input_dir, "**", f"*{ext}"), recursive=True)
        )
    print(f"[predict] 找到 {len(audio_files)} 个音频文件")

    results = []
    for i, audio_path in enumerate(audio_files):
        audio = load_audio(audio_path)
        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features.to(DEVICE)

        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred_ids = model.generate(inputs, **gen_kwargs)
            else:
                pred_ids = model.generate(inputs, **gen_kwargs)

        text = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )[0]

        results.append({
            "audio_path": os.path.basename(audio_path),
            "pred_text": text,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(audio_files)}] {time.strftime('%H:%M:%S')}")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[predict] 完成! 输出: {args.output} ({len(results)} 条)")


if __name__ == "__main__":
    main()
