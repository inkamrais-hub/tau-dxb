"""List all files in HF dataset repo."""
import sys
try:
    from huggingface_hub import list_repo_files
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
    from huggingface_hub import list_repo_files

files = list_repo_files('leeduckgo/cantonese-life-scenarios-corpus', repo_type='dataset')
for f in sorted(files):
    print(f)
