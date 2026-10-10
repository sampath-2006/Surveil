import numpy as np
import cv2
import os
import onnxruntime as ort
from src.utils.logger import logger

class DangerTrigger:
    """
    Lightweight MobileNetV3 binary classifier (Safe vs Suspect).
    Runs continuously on the Surveillance thread to gate the VLM.
    """
    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.session = None
        self.input_name = None
        
    def load(self):
        try:
            if os.path.exists(self.model_path):
                logger.info("Loading MobileNetV3 Trigger model (INT8 ONNX)...")
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                logger.info("Trigger model loaded successfully.")
            else:
                logger.warning(f"Trigger model {self.model_path} not found. Running in simulation mode.")
                self.session = "mock_session"
        except Exception as e:
            logger.error(f"Failed to load trigger model: {e}")
            
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize for MobileNetV3."""
        # Standard ImageNet normalization
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return img
        
    def infer(self, frame: np.ndarray) -> bool:
        """Returns True if danger is suspected."""
        if not self.session:
            return False
            
        if self.session == "mock_session":
            # MOCK IMPLEMENTATION: Randomly trigger ~2% of the time for testing
            prob_danger = np.random.rand() 
            return prob_danger > 0.98 

        input_tensor = self.preprocess(frame)
        
        # ONNX inference
        try:
            result = self.session.run(None, {self.input_name: input_tensor})
            # Assuming output shape is (1, num_classes) and class 1 is "danger"
            prob_danger = result[0][0][1] if result[0].shape[1] > 1 else result[0][0][0]
            is_suspect = prob_danger > self.confidence_threshold
            return is_suspect
        except Exception as e:
            logger.error(f"Inference error in trigger: {e}")
            return False
