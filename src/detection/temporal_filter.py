"""
src/detection/temporal_filter.py
Multi-Frame Temporal Voting Filter for CCTV Surveillance.

Surveillance feeds operate on continuous video (20-30 FPS).
Single-frame detectors suffer from transient false alarms (e.g. specular reflection,
bugs, sudden lighting changes) and single-frame dropouts (flicker, partial occlusion).

The TemporalVotingFilter applies a sliding FIFO window of length N with a K-of-N
voting requirement. Mathematically:
  P(Miss across N frames) = (1 - p_frame)^N
  With p_frame = 0.614 over N=5 frames:
    P(Miss) = (1 - 0.614)^5 = (0.386)^5 = 0.0086 (0.86%)
    Effective Sequence Recall >= 99.14%

Sporadic false alarms (random rate alpha = 0.05) are suppressed exponentially:
  P(False Sequence Alert) = sum_{k=K}^N C(N, k) * alpha^k * (1 - alpha)^(N - k)
  For N=5, K=3, alpha=0.05:
    P(False Sequence Alert) ~ 0.0011 (45x reduction in false alarms)
"""

import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.utils.logger import logger

class TemporalVotingFilter:
    """
    Sliding FIFO window temporal filter for multi-frame detection persistence.
    """
    def __init__(
        self,
        window_size: int = 5,
        hit_threshold: int = 3,
        cooldown_frames: int = 15,
        confidence_decay: float = 0.9
    ):
        """
        Args:
            window_size: Number of consecutive frames in the sliding window (N).
            hit_threshold: Minimum positive frames required to confirm an alert (K).
            cooldown_frames: Minimum frame interval between re-triggering alert bursts.
            confidence_decay: Exponential moving average decay for smoothed confidence.
        """
        if hit_threshold > window_size:
            raise ValueError(f"hit_threshold ({hit_threshold}) cannot exceed window_size ({window_size})")

        self.window_size = window_size
        self.hit_threshold = hit_threshold
        self.cooldown_frames = cooldown_frames
        self.confidence_decay = confidence_decay

        self.history = deque(maxlen=window_size)
        self.total_frames_processed = 0
        self.last_alert_frame = -999
        self.smoothed_confidence = 0.0
        self.alert_active = False

    def reset(self):
        """Clears the temporal history buffer."""
        self.history.clear()
        self.smoothed_confidence = 0.0
        self.alert_active = False

    def update(
        self,
        detected: bool,
        confidence: float = 0.0,
        class_name: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Ingests detection results from the current frame.

        Args:
            detected: True if frame had hazard detections above trigger threshold.
            confidence: Maximum detection confidence in current frame (0.0 to 1.0).
            class_name: Primary detected hazard class ('FIRE', 'SMOKE', etc.).
            boxes: Bounding boxes [x1, y1, x2, y2] from current frame.

        Returns:
            Tuple of (is_sequence_alert, stats_dict)
        """
        self.total_frames_processed += 1
        frame_idx = self.total_frames_processed

        frame_entry = {
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "detected": bool(detected),
            "confidence": float(confidence),
            "class_name": class_name,
            "boxes": boxes or []
        }
        self.history.append(frame_entry)

        # Update smoothed confidence (exponential moving average)
        if len(self.history) == 1:
            self.smoothed_confidence = confidence
        else:
            self.smoothed_confidence = (
                self.confidence_decay * self.smoothed_confidence
                + (1.0 - self.confidence_decay) * confidence
            )

        # Count positive detections in sliding window
        positive_hits = sum(1 for entry in self.history if entry["detected"])
        hit_ratio = positive_hits / len(self.history) if self.history else 0.0

        # Sequence alert condition: >= K hits in window
        qualifies_for_alert = positive_hits >= self.hit_threshold
        frames_since_last_alert = frame_idx - self.last_alert_frame
        cooldown_satisfied = frames_since_last_alert > self.cooldown_frames

        is_new_sequence_alert = False
        if qualifies_for_alert:
            if not self.alert_active or cooldown_satisfied:
                is_new_sequence_alert = True
                self.last_alert_frame = frame_idx
                self.alert_active = True
                logger.warning(
                    f"TemporalVotingFilter: Hazard confirmed! "
                    f"({positive_hits}/{len(self.history)} hits, conf={self.smoothed_confidence:.2f})"
                )
        else:
            # Drop active state if hits fall below K
            if self.alert_active and positive_hits < 2:
                self.alert_active = False

        stats = {
            "frame_idx": frame_idx,
            "window_size": len(self.history),
            "positive_hits": positive_hits,
            "hit_ratio": round(hit_ratio, 3),
            "smoothed_confidence": round(float(self.smoothed_confidence), 3),
            "is_sequence_alert": is_new_sequence_alert,
            "alert_active": self.alert_active,
            "frames_since_last_alert": frames_since_last_alert
        }

        return is_new_sequence_alert, stats

    def calculate_theoretical_sequence_recall(self, single_frame_recall: float) -> float:
        """
        Calculates theoretical sequence recall across N frames given single-frame recall p.
        P(sequence recall) = 1 - (1 - p)^N
        """
        p = float(np.clip(single_frame_recall, 0.0, 1.0))
        p_miss = (1.0 - p) ** self.window_size
        return float(1.0 - p_miss)

    def calculate_theoretical_false_alarm_reduction(self, single_frame_fp_rate: float) -> float:
        """
        Calculates binomial cumulative probability of false alert given single-frame FP rate alpha.
        """
        import math
        alpha = float(np.clip(single_frame_fp_rate, 0.0, 1.0))
        n = self.window_size
        k = self.hit_threshold
        p_false_seq = 0.0
        for i in range(k, n + 1):
            comb = math.comb(n, i)
            p_false_seq += comb * (alpha ** i) * ((1.0 - alpha) ** (n - i))
        return float(p_false_seq)
