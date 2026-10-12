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
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            if not self.cap.isOpened():
                logger.warning(f"Physical camera {self.camera_id} unavailable. Operating in synthetic live surveillance feed mode.")
                self.cap = None
        except Exception as e:
            logger.warning(f"Camera open error ({e}). Operating in synthetic live surveillance feed mode.")
            self.cap = None
            
    def stop(self):
        logger.info("Releasing camera")
        if self.cap:
            self.cap.release()
            self.cap = None

    def _generate_synthetic_frame(self, frame_count: int) -> np.ndarray:
        w, h = self.resolution
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Deep room background gradient
        for y in range(h):
            shade = int(25 + 30 * (y / h))
            img[y, :] = (shade, shade + 5, shade + 15)

        # Draw room geometry (walls, doorway, floor horizon)
        cv2.line(img, (0, int(h * 0.7)), (w, int(h * 0.7)), (45, 55, 65), 2)
        cv2.line(img, (int(w * 0.15), 0), (int(w * 0.15), int(h * 0.7)), (40, 50, 60), 2)
        cv2.line(img, (int(w * 0.85), 0), (int(w * 0.85), int(h * 0.7)), (40, 50, 60), 2)
        # Door frame
        cv2.rectangle(img, (int(w * 0.65), int(h * 0.25)), (int(w * 0.82), int(h * 0.7)), (55, 65, 80), 2)
        # Desk/table
        cv2.rectangle(img, (int(w * 0.2), int(h * 0.55)), (int(w * 0.55), int(h * 0.7)), (65, 75, 90), -1)
        cv2.rectangle(img, (int(w * 0.2), int(h * 0.55)), (int(w * 0.55), int(h * 0.7)), (90, 105, 125), 2)

        # Subtle simulated ambient motion / noise
        noise = (np.random.randn(h, w) * 3).astype(np.int16)
        img_noise = np.clip(img.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)

        # Surveillance telemetry overlays
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        ms = int((time.time() % 1) * 1000)
        cv2.putText(img_noise, f"REC [LIVE]  CAM-01 (ROOM 101)  {now_str}.{ms:03d}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
        cv2.putText(img_noise, f"FPS: {self.target_fps} | SSIM GATED | 640x480 INT8", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

        # Scanning target crosshair
        cx = int(w * 0.38 + np.sin(frame_count * 0.05) * 40)
        cy = int(h * 0.5 + np.cos(frame_count * 0.05) * 20)
        cv2.circle(img_noise, (cx, cy), 15, (0, 200, 255), 1)
        cv2.line(img_noise, (cx - 20, cy), (cx + 20, cy), (0, 200, 255), 1)
        cv2.line(img_noise, (cx, cy - 20), (cx, cy + 20), (0, 200, 255), 1)

        return img_noise
            
    def get_frames(self) -> Generator[Tuple[np.ndarray, bool], None, None]:
        """
        Yields (frame, is_keyframe) pairs.
        """
        self.start()
        frame_idx = 0
            
        while True:
            frame = None
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    frame = None
                    
            if frame is None:
                # Use synthetic live surveillance frame
                frame = self._generate_synthetic_frame(frame_idx)
                time.sleep(1.0 / self.target_fps)
                
            frame_idx += 1
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
                
            is_key = self.keyframe_selector.is_keyframe(frame)
            yield frame, is_key
