import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.utils.logger import logger

class KeyframeSelector:
    """
    Selects keyframes based on Structural Similarity Index (SSIM).
    Crucial for RPi5 to avoid processing identical static frames.
    """
    def __init__(self, ssim_threshold: float = 0.92, resize_dim: tuple = (320, 240)):
        self.ssim_threshold = ssim_threshold
        self.resize_dim = resize_dim
        self.last_keyframe_gray = None

    def is_keyframe(self, frame: np.ndarray) -> bool:
        """
        Determines if the current frame is significantly different from the last keyframe.
        """
        if frame is None:
            return False

        # Convert to grayscale and resize for fast SSIM calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, self.resize_dim)

        if self.last_keyframe_gray is None:
            self.last_keyframe_gray = small_gray
            return True

        # Calculate SSIM
        score, _ = ssim(self.last_keyframe_gray, small_gray, full=True)
        
        # If score is below threshold, scene has changed significantly
        if score < self.ssim_threshold:
            self.last_keyframe_gray = small_gray
            return True
            
        return False
