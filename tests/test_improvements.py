"""
tests/test_improvements.py
Unit and Integration Tests for Detection Improvements:
1. TemporalVotingFilter (Milestone 14)
2. CascadedDetector (Milestone 15)
3. SAHIDetector (Milestone 16)
"""

import pytest
import numpy as np
from src.detection.temporal_filter import TemporalVotingFilter
from src.detection.sahi_detector import SAHIDetector, nms_boxes
from src.detection.cascaded_detector import CascadedDetector

# ==============================================================================
# 1. TemporalVotingFilter Tests
# ==============================================================================

def test_temporal_filter_sliding_window():
    """Verify that temporal filter maintains sliding FIFO window of size N."""
    tf = TemporalVotingFilter(window_size=5, hit_threshold=3)
    assert tf.window_size == 5
    assert tf.hit_threshold == 3

    # Add 3 safe frames
    for _ in range(3):
        alert, stats = tf.update(detected=False, confidence=0.05)
        assert not alert
        assert not stats["is_sequence_alert"]

    assert stats["window_size"] == 3
    assert stats["positive_hits"] == 0

def test_temporal_filter_false_alarm_rejection():
    """Verify that isolated single-frame false alarm does NOT trigger alert."""
    tf = TemporalVotingFilter(window_size=5, hit_threshold=3)

    # Frame 1: Safe
    alert, _ = tf.update(detected=False)
    assert not alert

    # Frame 2: Transient false alarm (e.g. sunlight glare)
    alert, stats = tf.update(detected=True, confidence=0.88)
    assert not alert  # Must be rejected because 1 < 3
    assert stats["positive_hits"] == 1

    # Frame 3: Safe again
    alert, stats = tf.update(detected=False)
    assert not alert
    assert stats["positive_hits"] == 1

def test_temporal_filter_sequence_confirmation():
    """Verify that K-of-N consecutive hits trigger a confirmed sequence alert."""
    tf = TemporalVotingFilter(window_size=5, hit_threshold=3, cooldown_frames=10)

    # Ingest 3 hits in 4 frames
    tf.update(detected=True, confidence=0.7)
    tf.update(detected=False, confidence=0.1)
    tf.update(detected=True, confidence=0.75)
    alert, stats = tf.update(detected=True, confidence=0.8)

    assert alert is True
    assert stats["is_sequence_alert"] is True
    assert stats["alert_active"] is True
    assert stats["positive_hits"] == 3

def test_temporal_filter_math_predictions():
    """Verify the mathematical theoretical sequence recall and false alarm reduction."""
    tf = TemporalVotingFilter(window_size=5, hit_threshold=3)

    # Base single-frame recall: 0.614
    seq_recall = tf.calculate_theoretical_sequence_recall(0.614)
    # P(miss) = (1 - 0.614)^5 = (0.386)^5 = 0.0086 => Recall = 99.14%
    assert seq_recall > 0.99

    # Base single-frame FP rate: 0.05 (5%)
    fa_reduction = tf.calculate_theoretical_false_alarm_reduction(0.05)
    # Binomial sum for k>=3 out of 5 with p=0.05 is ~0.00115 (<0.002)
    assert fa_reduction < 0.002

# ==============================================================================
# 2. SAHIDetector Tests
# ==============================================================================

def test_sahi_slice_generation():
    """Verify that SAHI generates covering slices across frame dimensions."""
    sahi = SAHIDetector(slice_height=200, slice_width=200, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    slices = sahi.generate_slices((400, 400))

    assert len(slices) >= 4
    # All slices must be within image boundaries
    for (x1, y1, x2, y2) in slices:
        assert 0 <= x1 < x2 <= 400
        assert 0 <= y1 < y2 <= 400
        assert (x2 - x1) <= 200
        assert (y2 - y1) <= 200

def test_nms_boxes():
    """Verify Non-Maximum Suppression suppresses overlapping redundant boxes."""
    boxes = [
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 52.0, 52.0],  # Heavy overlap with box 0
        [100.0, 100.0, 150.0, 150.0] # Distinct box
    ]
    scores = [0.90, 0.75, 0.85]

    keep = nms_boxes(boxes, scores, iou_threshold=0.4)
    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep  # Box 1 suppressed by Box 0

# ==============================================================================
# 3. CascadedDetector Tests
# ==============================================================================

def test_cascaded_candidate_partition():
    """Verify partitioning of detections into confirmed vs suspicious."""
    cascade = CascadedDetector(low_conf_threshold=0.10, high_conf_threshold=0.65)

    detections = [
        {"name": "fire", "conf": 0.82, "box": [10, 10, 50, 50]},  # High conf -> confirmed
        {"name": "smoke", "conf": 0.35, "box": [20, 20, 80, 80]}, # Mid conf -> suspect
        {"name": "fire", "conf": 0.05, "box": [0, 0, 10, 10]}     # Sub-threshold -> dropped
    ]

    confirmed, suspicious = cascade.partition_candidates(detections)
    assert len(confirmed) == 1
    assert confirmed[0]["conf"] == 0.82
    assert len(suspicious) == 1
    assert suspicious[0]["conf"] == 0.35

def test_cascaded_crop_extraction():
    """Verify contextual margin padding around bounding box crop."""
    cascade = CascadedDetector(crop_padding_ratio=0.20)
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    box = [50.0, 50.0, 100.0, 100.0] # 50x50 box, 20% pad is 10px each side -> [40, 40, 110, 110]
    crop = cascade.extract_crop(img, box)

    assert crop.shape[0] == 70  # 110 - 40
    assert crop.shape[1] == 70  # 110 - 40
