"""
src/detection/sahi_detector.py
Slicing Aided Hyper Inference (SAHI) for Small and Distant Hazard Detection.

In high-mounted surveillance cameras (1080p / 4K), small or distant flames
(e.g., distant candle, electrical spark, early-stage fire) occupy only 16x16 to
32x32 pixels in the raw image. When downsampled to 640x640 for edge detector input,
these tiny features blur into unrecognizable smudges, causing severe false negatives (Recall < 50%).

SAHI partitions the full-resolution frame into overlapping slices/tiles, runs
inference on each slice at native pixel density, maps bounding boxes back to global
coordinates, and applies Non-Maximum Suppression (NMS) with full-frame detections.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from src.utils.logger import logger

def nms_boxes(boxes: List[List[float]], scores: List[float], iou_threshold: float = 0.45) -> List[int]:
    """
    Standard Non-Maximum Suppression (NMS) for bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    Returns indices of retained boxes.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

class SAHIDetector:
    """
    Slicing Aided Hyper Inference wrapper for edge vision models.
    """
    def __init__(
        self,
        slice_height: int = 384,
        slice_width: int = 384,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        iou_threshold: float = 0.45
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.iou_threshold = iou_threshold

    def generate_slices(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Calculates overlapping slice bounding coordinates (x1, y1, x2, y2).
        Handles image boundary clamping.
        """
        h_img, w_img = image_shape[:2]
        step_y = int(self.slice_height * (1.0 - self.overlap_height_ratio))
        step_x = int(self.slice_width * (1.0 - self.overlap_width_ratio))

        slices = []
        y = 0
        while y < h_img:
            y_end = min(y + self.slice_height, h_img)
            y_start = max(0, y_end - self.slice_height)

            x = 0
            while x < w_img:
                x_end = min(x + self.slice_width, w_img)
                x_start = max(0, x_end - self.slice_width)

                slices.append((x_start, y_start, x_end, y_end))

                if x_end >= w_img:
                    break
                x += step_x

            if y_end >= h_img:
                break
            y += step_y

        return slices

    def predict_with_sahi(
        self,
        model,
        image: np.ndarray,
        conf_threshold: float = 0.15,
        include_full_image: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Runs SAHI sliced inference + optional full-image inference and performs global NMS.

        Args:
            model: Object detector with .predict(crop, conf=...) method (e.g. YOLO).
            image: Full-frame BGR numpy image.
            conf_threshold: Minimum detection confidence score.
            include_full_image: Whether to also run inference on the full scaled frame.

        Returns:
            List of merged detection dicts: [{'box': [x1, y1, x2, y2], 'conf': float, 'class_id': int, 'name': str}]
        """
        h_img, w_img = image.shape[:2]
        all_candidate_boxes = []
        all_candidate_scores = []
        all_candidate_classes = []
        all_candidate_names = []

        # 1. Full Image Inference (Captures large/global hazards)
        if include_full_image:
            try:
                full_res = model.predict(image, conf=conf_threshold, verbose=False)
                for r in full_res:
                    for b in r.boxes:
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        conf = float(b.conf[0].cpu().numpy())
                        cid = int(b.cls[0].cpu().numpy())
                        cname = model.names.get(cid, str(cid)) if hasattr(model, 'names') else str(cid)

                        all_candidate_boxes.append(xyxy)
                        all_candidate_scores.append(conf)
                        all_candidate_classes.append(cid)
                        all_candidate_names.append(cname)
            except Exception as e:
                logger.error(f"SAHI full image inference failed: {e}")

        # 2. Sliced Tile Inference (Captures tiny/distant hazards)
        slices = self.generate_slices((h_img, w_img))
        for (sx1, sy1, sx2, sy2) in slices:
            tile = image[sy1:sy2, sx1:sx2]
            if tile.size == 0:
                continue

            try:
                tile_res = model.predict(tile, conf=conf_threshold, verbose=False)
                for r in tile_res:
                    for b in r.boxes:
                        # Local coordinates within tile
                        tx1, ty1, tx2, ty2 = b.xyxy[0].cpu().numpy().tolist()
                        conf = float(b.conf[0].cpu().numpy())
                        cid = int(b.cls[0].cpu().numpy())
                        cname = model.names.get(cid, str(cid)) if hasattr(model, 'names') else str(cid)

                        # Shift to global image coordinates
                        gx1 = float(np.clip(tx1 + sx1, 0, w_img))
                        gy1 = float(np.clip(ty1 + sy1, 0, h_img))
                        gx2 = float(np.clip(tx2 + sx1, 0, w_img))
                        gy2 = float(np.clip(ty2 + sy1, 0, h_img))

                        all_candidate_boxes.append([gx1, gy1, gx2, gy2])
                        all_candidate_scores.append(conf)
                        all_candidate_classes.append(cid)
                        all_candidate_names.append(cname)
            except Exception as e:
                logger.error(f"SAHI slice inference failed at ({sx1},{sy1}): {e}")

        if not all_candidate_boxes:
            return []

        # 3. Class-aware Non-Maximum Suppression (NMS)
        final_detections = []
        unique_classes = set(all_candidate_classes)

        for target_cls in unique_classes:
            cls_indices = [i for i, c in enumerate(all_candidate_classes) if c == target_cls]
            cls_boxes = [all_candidate_boxes[i] for i in cls_indices]
            cls_scores = [all_candidate_scores[i] for i in cls_indices]

            keep_idx = nms_boxes(cls_boxes, cls_scores, self.iou_threshold)

            for k in keep_idx:
                orig_idx = cls_indices[k]
                final_detections.append({
                    "box": all_candidate_boxes[orig_idx],
                    "conf": round(all_candidate_scores[orig_idx], 3),
                    "class_id": all_candidate_classes[orig_idx],
                    "name": all_candidate_names[orig_idx]
                })

        # Sort descending by confidence
        final_detections.sort(key=lambda d: d["conf"], reverse=True)
        return final_detections
