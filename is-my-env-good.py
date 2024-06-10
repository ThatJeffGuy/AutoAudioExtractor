import torch
import torchaudio
import torchvision
import speechbrain

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("torchaudio version:", torchaudio.__version__)
print("torchvision version:", torchvision.__version__)
print("speechbrain version:", speechbrain.__version__)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))