import torch

def test_cuda():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("CUDNN version:", torch.backends.cudnn.version())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    test_cuda()
