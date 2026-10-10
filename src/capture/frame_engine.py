import cv2
import time
from typing import Generator, Tuple
import numpy as np
from src.utils.logger import logger
from src.capture.keyframe_selector import KeyframeSelector

class FrameEngine:
    """
    Handles camera capture and preprocessing.
    Designed to run efficiently in the Surveillance loop.
    """
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30, ssim_threshold: float = 0.92):
        self.camera_id = camera_id
        self.resolution = resolution
        self.target_fps = fps
        self.cap = None
        self.keyframe_selector = KeyframeSelector(ssim_threshold=ssim_threshold)
        
    def start(self):
        logger.info("Initializing camera", camera_id=self.camera_id)
        # In a real RPi environment with CSI, you might use 'nvarguscamerasrc' or similar via gstreamer.
        # Defaulting to standard V4L2 backend for USB cameras.
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if not self.cap.isOpened():
            logger.error("Failed to open camera", camera_id=self.camera_id)
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
    def stop(self):
        logger.info("Releasing camera")
        if self.cap:
            self.cap.release()
            
    def get_frames(self) -> Generator[Tuple[np.ndarray, bool], None, None]:
        """
        Yields (frame, is_keyframe) pairs.
        """
        if not self.cap or not self.cap.isOpened():
            self.start()
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame")
                time.sleep(0.1)
                continue
                
            # Resize to target resolution if camera didn't honor the setting
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
                
            is_key = self.keyframe_selector.is_keyframe(frame)
            yield frame, is_key
