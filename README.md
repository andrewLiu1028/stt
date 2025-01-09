# 語音轉文字應用程式

這是一個基於 OpenAI Whisper 模型的語音轉文字應用程式，能夠將各種音訊格式轉換為文字，並支援時間戳記標註。本程式支援 GPU 加速處理，並提供多種輸出格式選擇。

## 功能特點

- 支援多種音訊格式：MP3、WAV、M4A、FLAC
- 多種輸出格式：
  - 純文字逐字稿 (.txt)
  - 帶時間戳記的文字檔 (_with_time.txt)
  - SRT 字幕格式檔案 (.srt)
- 支援 GPU 加速處理
- 支援批次處理整個目錄的音訊檔案
- 自動偵測運行環境（GPU/CPU）
- 提供詳細的處理進度和記錄
- 使用繁體中文輸出

## 系統需求

### 基本需求
- Python 3.8 或更高版本
- FFmpeg（用於音訊處理）
- CUDA 相容的 NVIDIA 顯示卡（選配，用於 GPU 加速）

### CUDA 環境配置（選配）
如需啟用 GPU 加速，請確保：
1. 已安裝 NVIDIA 顯示卡驅動程式
2. 已安裝 CUDA Toolkit
3. 已安裝支援 CUDA 的 PyTorch 版本

## 安裝步驟

1. 安裝系統依賴：
```bash
# Ubuntu/Debian
apt-get update && apt-get install -y ffmpeg

# Windows
# 請從 FFmpeg 官方網站下載並安裝
```

2. 建立虛擬環境（建議）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安裝必要套件：

方法一：使用 requirements.txt 安裝
```bash
# 更新 pip
pip install --upgrade pip

# 安裝所有依賴套件
pip install -r requirements.txt
```

方法二：手動安裝各個套件
```bash
# 更新 pip
pip install --upgrade pip

# 安裝 PyTorch (支援 CUDA)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安裝其他必要套件
pip install openai-whisper tqdm colorama requests
```

## 使用說明

### 1. 檢查 CUDA 環境（選配）
運行環境檢查程式：
```bash
python gpu_check.py
```

### 2. 下載模型
首次使用前，請下載必要的模型檔案：
```bash
python model_download.py
```

### 3. 運行程式
執行主程式：
```bash
python app.py
```

程式會提示您輸入音訊檔案或目錄的路徑，支援以下兩種處理模式：
- 單一檔案處理：輸入單一音訊檔案的路徑
- 批次處理：輸入含有多個音訊檔案的目錄路徑

### 輸出檔案說明

程式會為每個處理的音訊檔案產生三種輸出：

1. `檔名.txt`
   - 純文字逐字稿
   - 不含時間戳記
   - 適合閱讀和編輯

2. `檔名_with_time.txt`
   - 帶時間戳記的文字檔
   - 格式：`[時:分:秒 - 時:分:秒] 文字內容`
   - 適合需要時間參考的場合

3. `檔名.srt`
   - SRT 格式字幕檔
   - 格式：`時:分:秒 : 時:分:秒 | 文字內容`
   - 適合用於影片字幕

## 效能考量

- GPU 模式下可顯著提升處理速度
- 處理時間與音訊檔案長度及品質成正比
- 建議使用較高品質的音訊輸入以提高識別準確度

## 故障排除

### CUDA 相關問題
如果遇到 CUDA 相關錯誤：
1. 執行 `gpu_check.py` 檢查 CUDA 環境
2. 確認是否已正確安裝 NVIDIA 驅動程式
3. 確認 PyTorch 版本是否支援當前的 CUDA 版本

### 模型載入問題
如果模型載入失敗：
1. 檢查網路連接狀態
2. 確認磁碟空間是否充足
3. 嘗試重新執行 `model_download.py`

### 音訊處理問題
如果音訊處理失敗：
1. 確認音訊檔案格式是否支援
2. 檢查 FFmpeg 是否正確安裝
3. 確認檔案是否完整且可正常播放

