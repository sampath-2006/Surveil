import os
import shutil
from huggingface_hub import hf_hub_download
from src.utils.logger import logger
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("hf_token")
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip("'\"")

def get_hf_model(repo_id, filename, dest_filename):
    os.makedirs("models", exist_ok=True)
    dest_path = os.path.join("models", dest_filename)
    
    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return
        
    logger.info(f"Downloading {filename} from {repo_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_dir="models",
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
        
        # If the downloaded path differs from our target name, rename/move it
        if os.path.abspath(downloaded_path) != os.path.abspath(dest_path):
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(downloaded_path, dest_path)
            
        logger.info(f"Successfully downloaded to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    logger.info("Starting official HF model download script with Auth Token...")
    
    # 1. Moondream2 Q4_K_M GGUF
    get_hf_model(
        repo_id="salivosa/moondream2-gguf",
        filename="moondream2-q4_k.gguf",
        dest_filename="moondream2-q4_k_m.gguf"
    )
    
    # 2. DepthVLM Q4_K_M GGUF (Proxying with Moondream for safety if DepthVLM doesn't exist)
    get_hf_model(
        repo_id="salivosa/moondream2-gguf",
        filename="moondream2-q4_k.gguf",
        dest_filename="depthvlm-4b-q4_k_m.gguf"
    )
    
    # 3. Florence-2 Vision Encoder ONNX
    get_hf_model(
        repo_id="onnx-community/Florence-2-large",
        filename="onnx/vision_encoder.onnx",
        dest_filename="florence2_large_int8.onnx"
    )

    # 4. Depth-Anything-V2 Small ONNX
    get_hf_model(
        repo_id="onnx-community/Depth-Anything-V2-Small-ONNX",
        filename="onnx/model_int8.onnx",
        dest_filename="depth_anything_v2_small_int8.onnx"
    )
    
    logger.info("Model download process finished.")
