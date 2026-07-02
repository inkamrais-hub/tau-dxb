import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
