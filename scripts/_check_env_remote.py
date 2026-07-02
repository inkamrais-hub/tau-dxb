import importlib, sys
pkgs = ["triton", "transformers", "jiwer", "soundfile", "librosa", "accelerate", "torch", "torchaudio", "numpy"]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, "__version__", "ok")
        print(f"{p}: {ver}")
    except Exception as e:
        print(f"{p}: MISSING ({e})")
print("python:", sys.version.split()[0])
