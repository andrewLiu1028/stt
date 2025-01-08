def check_cuda_availability():
    """檢查CUDA環境"""
    import torch
    import sys
    import platform
    from colorama import Fore, Style
    
    print(f"\n{Fore.CYAN}=== CUDA 環境檢查 ==={Style.RESET_ALL}")
    print(f"作業系統: {platform.system()} {platform.version()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 檢查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
        print(f"GPU 數量: {torch.cuda.device_count()}")
    else:
        print(f"\n{Fore.RED}CUDA 不可用的可能原因：{Style.RESET_ALL}")
        print("1. 未安裝 NVIDIA 顯示卡驅動程式")
        print("2. 未安裝 CUDA Toolkit")
        print("3. PyTorch 未安裝 CUDA 版本")
        print("\n解決方法：")
        print("1. 確認是否已安裝最新的 NVIDIA 顯示卡驅動程式")
        print("2. 安裝 CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)")
        print("3. 重新安裝 PyTorch CUDA 版本：")
        print(f"{Fore.GREEN}pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118{Style.RESET_ALL}")
    
    return cuda_available

if __name__ == "__main__":
    check_cuda_availability()