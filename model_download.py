import whisper
import torch
from pathlib import Path
import logging
import colorama
from colorama import Fore, Style

# 設定基本配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
colorama.init()

def download_model():
    try:
        # 設定模型存放路徑
        base_dir = Path(__file__).parent
        models_dir = base_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # 顯示設備信息
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{'='*50}")
        if device == "cuda":
            print(f"{Fore.GREEN}使用 GPU 進行處理{Style.RESET_ALL}")
            print(f"GPU型號: {torch.cuda.get_device_name()}")
            print(f"可用GPU數量: {torch.cuda.device_count()}")
            print(f"當前GPU記憶體使用: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        else:
            print(f"{Fore.YELLOW}使用 CPU 進行處理{Style.RESET_ALL}")
        print(f"{'='*50}\n")

        # 下載並載入模型
        logging.info(f"{Fore.CYAN}開始下載和載入模型...{Style.RESET_ALL}")
        model = whisper.load_model(
            name="large-v3-turbo",
            device=device,
            download_root=str(models_dir)
        )

        if model is not None:
            logging.info(f"{Fore.GREEN}模型載入成功！{Style.RESET_ALL}")
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated()/1024**2
                logging.info(f"GPU記憶體使用: {memory_used:.1f} MB")
            return True
    except Exception as e:
        logging.error(f"{Fore.RED}模型下載或載入失敗: {str(e)}{Style.RESET_ALL}")
        return False

if __name__ == "__main__":
    print(f"{Fore.CYAN}=== Whisper 模型下載程式 ==={Style.RESET_ALL}")
    if download_model():
        print(f"\n{Fore.GREEN}模型已成功下載到 'models' 目錄{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}模型下載失敗{Style.RESET_ALL}")