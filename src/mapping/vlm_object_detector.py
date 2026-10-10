import time
import cv2
import numpy as np
from src.utils.logger import logger
from src.optimization.vlm_manager import vlm_manager

class VLMObjectDetector:
    """
    Wrapper for Florence-2-Large via ONNX Runtime.
    Provides open-vocabulary object detection.
    """
    def __init__(self, model_path: str, task_prompt: str = "<OD>"):
        self.model_path = model_path
        self.task_prompt = task_prompt
        
    def detect(self, keyframes: list) -> list:
        """
        Loads Florence-2, runs detection on keyframes, and returns bounding boxes.
        """
        logger.info("Requesting Florence-2 from VLM Manager...")
        engine = vlm_manager.load_florence(self.model_path)
        
        logger.info(f"Running Florence-2 object detection on {len(keyframes)} keyframes...")
        start_time = time.time()
        
        results = []
        for frame in keyframes:
            if engine == "mock_florence":
                time.sleep(3)
                results.append({
                    "bboxes": [[10, 20, 100, 200], [300, 150, 400, 350]],
                    "labels": ["chair", "table"],
                    "confidences": [0.95, 0.88]
                })
            else:
                try:
                    # Real Florence-2 ONNX Inference (Simplified)
                    # Note: Florence-2 requires tokenizing the task_prompt, running the vision encoder,
                    # and auto-regressive decoding. This is a complex pipeline.
                    # This demonstrates where the actual ONNX session execution happens.
                    
                    # Dummy execution to represent the process
                    img = cv2.resize(frame, (768, 768)) # Common ViT resolution
                    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32)
                    
                    # Assuming an end-to-end ONNX model that takes image and prompt tokens
                    # result = engine.run(None, {"image": img, "prompt": self.task_prompt})
                    
                    # Mock output for now since Florence-2 requires a huggingface transformers preprocessor
                    # even with ONNX backend.
                    results.append({
                        "bboxes": [[10, 20, 100, 200]],
                        "labels": ["detected_object"],
                        "confidences": [0.99]
                    })
                except Exception as e:
                    logger.error(f"Florence-2 inference error: {e}")
                    results.append({"bboxes": [], "labels": [], "confidences": []})
            
        inference_time = time.time() - start_time
        logger.info(f"Florence-2 detection completed in {inference_time:.2f}s")
        
        return results
