import os
import requests
from pathlib import Path
from src.utils.logger import logger

MODELS = {
    "mobilenetv3_small_int8.onnx": "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx", # Placeholder public ONNX
    "moondream2-q4_k_m.gguf": "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-4bit.gguf", # Example link
    "florence2_large_int8.onnx": "https://huggingface.co/microsoft/Florence-2-large/resolve/main/config.json", # Needs proper export to ONNX in practice
    "depthvlm-4b-q4_k_m.gguf": "https://huggingface.co/vikhyatk/moondream2/resolve/main/moondream2-text-model-4bit.gguf" # Placeholder for DepthVLM
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return
        
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        # Stream download to avoid RAM bloat
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        logger.info(f"Download complete: {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")

def main():
    logger.info("Starting model download script...")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, url in MODELS.items():
        dest = models_dir / filename
        download_file(url, str(dest))
        
    logger.info("Model download process finished. Note: Some links are placeholders for ONNX/GGUF exports.")

if __name__ == "__main__":
    main()
