"""Quick find FFN class name - lightweight version."""
import transformers.models.whisper.modeling_whisper as w
names = [x for x in dir(w) if "FFN" in x or "Feed" in x or "MLP" in x or "Dense" in x]
print("FFN-related names:", names)
