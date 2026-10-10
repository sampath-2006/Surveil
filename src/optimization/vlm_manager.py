import time
import threading
import os
from src.utils.logger import logger

class VLMManager:
    """
    Manages the sequential loading and unloading of VLMs to adhere to the 
    strict memory limits of the Raspberry Pi 5.
    Ensures only one major VLM is loaded in memory at a time.
    """
    def __init__(self):
        self._current_model_name = None
        self._current_engine = None
        self._lock = threading.Lock()
        
    def _unload_current(self):
        if self._current_engine is not None:
            logger.info(f"Unloading model: {self._current_model_name} from memory")
            del self._current_engine
            self._current_engine = None
            self._current_model_name = None
            import gc
            gc.collect()

    def load_moondream(self, model_path: str):
        """Loads Moondream 2B for danger analysis."""
        with self._lock:
            if self._current_model_name == "moondream":
                return self._current_engine
                
            self._unload_current()
            logger.info(f"Loading Moondream 2B from {model_path}...")
            
            if os.path.exists(model_path):
                from llama_cpp import Llama
                self._current_engine = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose=False)
            else:
                logger.warning(f"Model file {model_path} not found. Using mock engine.")
                self._current_engine = "mock_moondream"
                
            self._current_model_name = "moondream"
            logger.info("Moondream 2B ready")
            return self._current_engine

    def load_florence(self, model_path: str):
        """Loads Florence-2 for object detection mapping."""
        with self._lock:
            if self._current_model_name == "florence":
                return self._current_engine
                
            self._unload_current()
            logger.info(f"Loading Florence-2 from {model_path}...")
            
            if os.path.exists(model_path):
                import onnxruntime as ort
                self._current_engine = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            else:
                logger.warning(f"Model file {model_path} not found. Using mock engine.")
                self._current_engine = "mock_florence"
                
            self._current_model_name = "florence"
            logger.info("Florence-2 ready")
            return self._current_engine

    def load_depthvlm(self, model_path: str):
        """Loads DepthVLM for dense metric depth mapping."""
        with self._lock:
            if self._current_model_name == "depthvlm":
                return self._current_engine
                
            self._unload_current()
            logger.info(f"Loading DepthVLM from {model_path}...")
            
            if os.path.exists(model_path):
                from llama_cpp import Llama
                self._current_engine = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose=False)
            else:
                logger.warning(f"Model file {model_path} not found. Using mock engine.")
                self._current_engine = "mock_depthvlm"
                
            self._current_model_name = "depthvlm"
            logger.info("DepthVLM ready")
            return self._current_engine
            
    def unload_all(self):
        with self._lock:
            self._unload_current()

# Global instance
vlm_manager = VLMManager()
