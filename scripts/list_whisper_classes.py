"""Find ALL class names in whisper modeling for debugging."""
import transformers.models.whisper.modeling_whisper as w
all_names = sorted([x for x in dir(w) if not x.startswith('_')])
print("All public names:", all_names)
