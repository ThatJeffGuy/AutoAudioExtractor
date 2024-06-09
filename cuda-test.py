import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
