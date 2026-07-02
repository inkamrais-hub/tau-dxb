"""Find the correct FFN/MLP class name in Whisper model."""
import transformers.models.whisper.modeling_whisper as w
names = [x for x in dir(w) if "FFN" in x or "Feed" in x or "MLP" in x or "Dense" in x or "dense" in x or "intermediate" in x]
print("Matching names:", names)
print()

# Also check the WhisperEncoderLayer and WhisperDecoderLayer structure
import torch
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir="/hf_cache")
print("Encoder layer children:")
for name, module in model.model.encoder.layers[0].named_children():
    print(f"  {name}: {module.__class__.__name__}")
print("Decoder layer children:")
for name, module in model.model.decoder.layers[0].named_children():
    print(f"  {name}: {module.__class__.__name__}")
