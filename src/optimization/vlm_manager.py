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
        
    def _resolve_path(self, path: str) -> str:
        if os.path.exists(path):
            return path
        candidate = os.path.join("models", path)
        if os.path.exists(candidate):
            return candidate
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate_abs = os.path.join(base_dir, "models", os.path.basename(path))
        if os.path.exists(candidate_abs):
            return candidate_abs
        return path

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
            resolved = self._resolve_path(model_path)
            logger.info(f"Loading Moondream 2B from {resolved}...")
            
            if os.path.exists(resolved):
                try:
                    from llama_cpp import Llama
                    self._current_engine = Llama(model_path=resolved, n_ctx=2048, n_threads=2, verbose=False)
                except Exception as e:
                    logger.warning(f"Could not initialize llama-cpp for {resolved}: {e}. Using mock engine.")
                    self._current_engine = "mock_moondream"
            else:
                logger.warning(f"Model file {resolved} not found. Using mock engine.")
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
            resolved = self._resolve_path(model_path)
            logger.info(f"Loading Florence-2 from {resolved}...")
            
            if os.path.exists(resolved):
                try:
                    import onnxruntime as ort
                    self._current_engine = ort.InferenceSession(resolved, providers=['CPUExecutionProvider'])
                except Exception as e:
                    logger.warning(f"Could not load Florence-2 ONNX from {resolved}: {e}. Using simulation engine.")
                    self._current_engine = "mock_florence"
            else:
                logger.warning(f"Model file {resolved} not found. Using mock engine.")
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
            resolved = self._resolve_path(model_path)
            logger.info(f"Loading DepthVLM from {resolved}...")
            
            if os.path.exists(resolved):
                try:
                    from llama_cpp import Llama
                    self._current_engine = Llama(model_path=resolved, n_ctx=2048, n_threads=2, verbose=False)
                except Exception as e:
                    logger.warning(f"Could not load DepthVLM from {resolved}: {e}. Using simulation engine.")
                    self._current_engine = "mock_depthvlm"
            else:
                logger.warning(f"Model file {resolved} not found. Using mock engine.")
                self._current_engine = "mock_depthvlm"
                
            self._current_model_name = "depthvlm"
            logger.info("DepthVLM ready")
            return self._current_engine
            
    def unload_all(self):
        with self._lock:
            self._unload_current()

# Global instance
vlm_manager = VLMManager()
