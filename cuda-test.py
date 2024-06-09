import torch

def check_cuda():
    try:
        import torch
        if not hasattr(torch, 'cuda'):
            raise ImportError("CUDA module is not available in torch")
        if not torch.cuda.is_available():
            raise ImportError("CUDA is not available on this system")
        else:
            print("CUDA is available. Device count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"CUDA check failed: {e}")
        sys.exit(1)

def check_cudnn():
    try:
        if torch.backends.cudnn.is_available():
            print("cuDNN is available")
        else:
            raise ImportError("cuDNN is not available")
    except Exception as e:
        print(f"cuDNN check failed: {e}")
        sys.exit(1)

check_cuda()
check_cudnn()
