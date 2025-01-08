import whisper
import torch
import warnings
from pathlib import Path
import logging
from tqdm import tqdm
from typing import List
import colorama
from colorama import Fore, Style

class AudioConverter:
   def __init__(self):
       self.model = None
       self.device = "cuda" if torch.cuda.is_available() else "cpu"
       self.models_dir = Path(__file__).parent / "models"
       self.models_dir.mkdir(exist_ok=True)
       
       colorama.init()
       self._print_device_info()
       
       logging.basicConfig(
           level=logging.INFO,
           format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
           handlers=[logging.StreamHandler()]
       )
       
       warnings.filterwarnings("ignore")
       
   def _print_device_info(self) -> None:
       device_info = f"\n{'='*50}\n"
       if self.device == "cuda":
           device_info += f"{Fore.GREEN}使用 GPU 進行處理{Style.RESET_ALL}\n"
           device_info += f"GPU型號: {torch.cuda.get_device_name()}\n"
           device_info += f"可用GPU數量: {torch.cuda.device_count()}\n"
           device_info += f"當前GPU記憶體使用: {torch.cuda.memory_allocated()/1024**2:.1f} MB\n"
       else:
           device_info += f"{Fore.YELLOW}使用 CPU 進行處理{Style.RESET_ALL}\n"
       device_info += f"{'='*50}"
       print(device_info)

   def load_model(self, model_name: str = "large-v3-turbo") -> bool:
       try:
           if self.model is None:
               logging.info(f"載入模型: {model_name}")
               
               # 指定本地模型路徑和名稱
               self.model = whisper.load_model(
                   name=model_name,
                   device=self.device,
                   download_root=str(self.models_dir)
               )
               
               if self.model is not None:
                   logging.info(f"{Fore.GREEN}模型載入成功！{Style.RESET_ALL}")
                   if self.device == "cuda":
                       memory_used = torch.cuda.memory_allocated()/1024**2
                       logging.info(f"GPU記憶體使用: {memory_used:.1f} MB")
                   return True
                   
       except Exception as e:
           logging.error(f"{Fore.RED}模型載入失敗: {str(e)}{Style.RESET_ALL}")
           return False

   def format_timestamp(self, seconds: float) -> str:
       hours = int(seconds // 3600)
       minutes = int((seconds % 3600) // 60)
       seconds = int(seconds % 60)
       return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

   def save_as_plain_txt(self, segments: List[dict], output_path: Path):
       """Save segments as plain text without timestamps or segments."""
       with open(output_path, 'w', encoding='utf-8') as f:
           full_text = ' '.join([segment['text'].strip() for segment in segments])
           f.write(full_text)

   def save_as_txt_with_timestamps(self, segments: List[dict], output_path: Path):
       """Save segments as plain text with timestamps in HH:MM:SS format."""
       with open(output_path, 'w', encoding='utf-8') as f:
           for segment in segments:
               start_time = self.format_timestamp(segment['start'])
               end_time = self.format_timestamp(segment['end'])
               text = segment['text']
               f.write(f"[{start_time} - {end_time}] {text}\n")

   def save_as_srt(self, segments: List[dict], output_path: Path):
       """Save segments as SRT file with timestamps in HH:MM:SS format."""
       with open(output_path, 'w', encoding='utf-8') as f:
           for i, segment in enumerate(segments, start=1):
               start_time = self.format_timestamp(segment['start'])
               end_time = self.format_timestamp(segment['end'])
               text = segment['text']
               f.write(f"{start_time} : {end_time} | {text}\n")

   def convert_audio(self, input_path: Path) -> bool:
       """Convert single audio file to text with timestamps."""
       try:
           if self.model is None:
               raise RuntimeError("模型尚未載入")
           if not input_path.exists():
               raise FileNotFoundError(f"找不到檔案: {input_path}")
           
           logging.info(f"處理檔案: {input_path}")
           result = self.model.transcribe(
               str(input_path), 
               language="zh", 
               fp16=self.device == "cuda",
               initial_prompt="以繁體中文轉錄內容，並確保用字正確:")
           
           segments = result.get('segments', [])
           if not segments:
               logging.warning("未檢測到有效內容")
               return False

           # 1. 純文字輸出（不帶時間戳記，不分段）
           output_txt_path = input_path.with_suffix('.txt')
           self.save_as_plain_txt(segments, output_txt_path)

           # 2. 帶時間戳記的 txt
           output_txt_with_time_path = input_path.with_name(input_path.stem + "_with_time.txt")
           self.save_as_txt_with_timestamps(segments, output_txt_with_time_path)

           # 3. 帶時間戳記的 srt
           output_srt_path = input_path.with_suffix('.srt')
           self.save_as_srt(segments, output_srt_path)

           logging.info(f"{Fore.GREEN}轉換完成！已產生以下檔案：{Style.RESET_ALL}")
           logging.info(f"1. 純文字逐字稿：{output_txt_path}")
           logging.info(f"2. 帶時間戳記的文字檔：{output_txt_with_time_path}")
           logging.info(f"3. SRT 格式檔案：{output_srt_path}")
           return True
           
       except Exception as e:
           logging.error(f"音檔轉換失敗: {str(e)}")
           return False

   def convert_directory(self, directory_path: Path) -> bool:
       """Convert all audio files in a directory."""
       try:
           if not directory_path.exists():
               raise NotADirectoryError(f"找不到目錄: {directory_path}")
           audio_extensions = {'.mp3', '.wav', '.m4a', '.flac'}
           audio_files = [f for f in directory_path.glob('**/*') if f.is_file() and f.suffix.lower() in audio_extensions]
           if not audio_files:
               logging.warning("在目錄中找不到音檔")
               return False

           logging.info(f"找到 {len(audio_files)} 個音檔")
           for audio_file in tqdm(audio_files, desc="轉換進度"):
               self.convert_audio(audio_file)
           return True
       except Exception as e:
           logging.error(f"目錄處理失敗: {str(e)}")
           return False

def main():
   print(f"\n{Fore.CYAN}=== 語音轉文字程式 ==={Style.RESET_ALL}")
   converter = AudioConverter()
   
   # 載入模型
   if not converter.load_model("large-v3-turbo"):
       print(f"{Fore.RED}模型載入失敗{Style.RESET_ALL}")
       return

   # 處理輸入路徑
   input_path = Path(input("請輸入音檔或目錄路徑: ").strip())
   if not input_path.exists():
       print(f"{Fore.RED}路徑無效{Style.RESET_ALL}")
       return

   # 處理檔案或目錄
   if input_path.is_file():
       converter.convert_audio(input_path)
   elif input_path.is_dir():
       converter.convert_directory(input_path)
   else:
       print(f"{Fore.RED}無效的路徑{Style.RESET_ALL}")

if __name__ == "__main__":
   main()