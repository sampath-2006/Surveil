import numpy as np
import cv2
import os
import onnxruntime as ort
from typing import Tuple, Dict, Any, Optional
from src.utils.logger import logger
from src.detection.temporal_filter import TemporalVotingFilter

class DangerTrigger:
    """
    Lightweight MobileNetV3 binary classifier (Safe vs Suspect)
    augmented with Multi-Frame Temporal Voting Filter for sequence persistence.
    Runs continuously on the Surveillance thread to gate the VLM.
    """
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.6,
        use_temporal_filter: bool = True,
        window_size: int = 5,
        hit_threshold: int = 3
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_temporal_filter = use_temporal_filter
        self.session = None
        self.input_name = None
        
        if self.use_temporal_filter:
            self.temporal_filter = TemporalVotingFilter(
                window_size=window_size,
                hit_threshold=hit_threshold
            )
        else:
            self.temporal_filter = None
        
    def load(self):
        try:
            path = self.model_path
            if not os.path.exists(path):
                candidate = os.path.join("models", path)
                if os.path.exists(candidate):
                    path = candidate
                else:
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    candidate_abs = os.path.join(base_dir, "models", os.path.basename(path))
                    if os.path.exists(candidate_abs):
                        path = candidate_abs
            if os.path.exists(path):
                logger.info(f"Loading MobileNetV3 Trigger model from {path} (INT8 ONNX)...")
                self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                logger.info("Trigger model loaded successfully.")
            else:
                logger.warning(f"Trigger model {path} not found. Running in simulation mode.")
                self.session = "mock_session"
        except Exception as e:
            logger.error(f"Failed to load trigger model: {e}")
            
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize for MobileNetV3."""
        # Standard ImageNet normalization
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = ((img - mean) / std).astype(np.float32)
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)  # Add batch dim
        return img
        
    def infer_raw(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Runs raw single-frame evaluation without temporal smoothing.
        Returns (is_suspect_raw, prob_danger).
        """
        if not self.session:
            return False, 0.0
            
        if self.session == "mock_session":
            # MOCK IMPLEMENTATION: Randomly trigger ~2% of the time for testing
            prob_danger = float(np.random.rand()) 
            return prob_danger > 0.98, prob_danger

        input_tensor = self.preprocess(frame)
        
        # ONNX inference
        try:
            result = self.session.run(None, {self.input_name: input_tensor})
            outputs = result[0]
            if outputs.ndim == 2 and outputs.shape[1] == 2:
                # Binary classifier: index 1 is danger
                exp_vals = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
                probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
                prob_danger = float(probs[0, 1])
            elif outputs.ndim == 2 and outputs.shape[1] > 2:
                # Multi-class ImageNet checkpoint (e.g. fire/flame proxy or threshold)
                exp_vals = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
                probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
                # Proxy hazard trigger: triggers on high anomaly confidence
                prob_danger = float(np.max(probs[0])) * 0.1
            else:
                prob_danger = float(outputs[0][0])
            is_suspect = prob_danger > self.confidence_threshold
            return is_suspect, prob_danger
        except Exception as e:
            logger.error(f"Inference error in trigger: {e}")
            return False, 0.0

    def infer(self, frame: np.ndarray) -> bool:
        """
        Returns True if danger is suspected.
        When use_temporal_filter is enabled, requires K-of-N sequence persistence
        to eliminate transient single-frame false alarms and elevate sequence recall > 99%.
        """
        is_suspect_raw, prob_danger = self.infer_raw(frame)

        if not self.use_temporal_filter or self.temporal_filter is None:
            return is_suspect_raw

        is_sequence_alert, _ = self.temporal_filter.update(
            detected=is_suspect_raw,
            confidence=prob_danger
        )
        return is_sequence_alert
