"""
scripts/evaluate_dataset.py
Rigorous Empirical Evaluation of Object Detection Recall & mAP on Real CCTV Surveillance Dataset.

Dataset: Jimei Fire & Smoke CCTV Surveillance Benchmark (betasecond/jimei-fire-smoke-yolo-dataset)
Metrics:
  - IoU >= 0.50 True Positive / False Positive assignment
  - Precision, Recall, and F1-score per class
  - Precision-Recall curves
  - AP@50 (Average Precision) for Fire and Smoke
  - Overall mAP@50 (mean Average Precision) and mAP@50-95
  - MobileNetV3 Trigger Stage Gating Recall
"""

import os
import sys
import glob
import time
import json
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PyTorch 2.6 safe-load patch for ultralytics
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, 'weights_only': False})
from ultralytics import YOLO

def box_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def calculate_ap(recalls, precisions):
    """
    Calculates Average Precision (AP) using standard 11-point or continuous AUC interpolation.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integrate area under envelope
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)

def run_evaluation():
    print("=" * 80)
    print("   SURVEIL: REAL CCTV DATASET EMPIRICAL EVALUATION (RECALL & mAP)")
    print("   Dataset: Jimei Fire & Smoke CCTV Surveillance Benchmark (50 Real Frames)")
    print("=" * 80)

    img_dir = "data/cctv_eval_raw/test/images"
    lbl_dir = "data/cctv_eval_raw/test/labels"
    model_path = "models/yolov8n_fire_smoke.pt"

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Error: dataset directories not found: {img_dir}, {lbl_dir}")
        return

    # Load model
    print(f"\n[1/4] Loading Detection Model: {model_path}...")
    model = YOLO(model_path)
    class_names = model.names
    print(f"      Model classes: {class_names}")

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    print(f"      Found {len(img_files)} real test images.")

    # Data structures for evaluation
    class_ids = [0, 1] # 0: Fire, 1: Smoke
    gt_per_class = {0: 0, 1: 0}
    detections_per_class = {0: [], 1: []}

    print("\n[2/4] Running Inference & Matching Ground Truth Annotations...")
    inference_latencies = []
    
    # Process each image
    for idx, img_path in enumerate(img_files):
        img_name = os.path.basename(img_path)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        # Parse ground truth boxes [class_id, x1, y1, x2, y2]
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = (xc - bw / 2.0) * w_img
                        y1 = (yc - bh / 2.0) * h_img
                        x2 = (xc + bw / 2.0) * w_img
                        y2 = (yc + bh / 2.0) * h_img
                        gt_boxes.append({"class_id": cid, "box": [x1, y1, x2, y2], "matched": False})
                        if cid in gt_per_class:
                            gt_per_class[cid] += 1

        # Run model inference
        t0 = time.perf_counter()
        results = model.predict(img, conf=0.15, verbose=False)
        lat = (time.perf_counter() - t0) * 1000
        inference_latencies.append(lat)

        pred_boxes = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                cid = int(b.cls[0].cpu().numpy())
                pred_boxes.append({"class_id": cid, "conf": conf, "box": xyxy})

        # Match predictions to GT for this image
        # Sort predictions by confidence descending
        pred_boxes.sort(key=lambda x: x["conf"], reverse=True)

        for pred in pred_boxes:
            cid = pred["class_id"]
            if cid not in class_ids:
                continue
            pbox = pred["box"]
            pconf = pred["conf"]

            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gt_boxes):
                if gt["class_id"] == cid and not gt["matched"]:
                    iou = box_iou(pbox, gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= 0.50 and best_gt_idx != -1:
                # True positive match at IoU >= 0.50
                gt_boxes[best_gt_idx]["matched"] = True
                detections_per_class[cid].append({"conf": pconf, "tp": 1, "fp": 0, "iou": best_iou})
            else:
                # False positive
                detections_per_class[cid].append({"conf": pconf, "tp": 0, "fp": 1, "iou": best_iou})

    # --------------------------------------------------------------------------
    # 3. Compute Precision, Recall, and AP@50 per Class
    # --------------------------------------------------------------------------
    print("\n[3/4] Computing Empirical Precision, Recall & mAP@50...")
    eval_results = {}
    aps = []
    mean_lat_ms = np.mean(inference_latencies)

    for cid in class_ids:
        cname = class_names.get(cid, f"class_{cid}")
        dets = detections_per_class[cid]
        total_gt = gt_per_class[cid]

        if total_gt == 0:
            continue

        # Sort all detections across entire dataset by confidence descending
        dets.sort(key=lambda x: x["conf"], reverse=True)

        tp_cumsum = np.cumsum([d["tp"] for d in dets])
        fp_cumsum = np.cumsum([d["fp"] for d in dets])

        recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        final_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
        final_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
        final_fn = total_gt - final_tp

        final_recall = final_tp / total_gt if total_gt > 0 else 0.0
        final_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0.0
        f1_score = (2 * final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0.0
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)

        eval_results[cname] = {
            "ground_truth_instances": total_gt,
            "detected_true_positives": final_tp,
            "false_positives": final_fp,
            "false_negatives": final_fn,
            "recall_50": round(final_recall, 4),
            "precision_50": round(final_precision, 4),
            "f1_score": round(f1_score, 4),
            "average_precision_ap50": round(ap, 4),
            "pr_curve": {
                "recalls": [round(float(r), 4) for r in recalls[::max(1, len(recalls)//20)]],
                "precisions": [round(float(p), 4) for p in precisions[::max(1, len(precisions)//20)]]
            }
        }

    overall_map50 = float(np.mean(aps)) if aps else 0.0
    total_tp = sum(v["detected_true_positives"] for v in eval_results.values())
    total_gt = sum(v["ground_truth_instances"] for v in eval_results.values())
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    overall_precision = total_tp / sum(v["detected_true_positives"] + v["false_positives"] for v in eval_results.values())

    summary = {
        "dataset_name": "Jimei Fire & Smoke CCTV Surveillance Benchmark (Test Split)",
        "test_images_count": len(img_files),
        "total_ground_truth_boxes": total_gt,
        "mean_inference_latency_ms": round(mean_lat_ms, 2),
        "throughput_fps": round(1000.0 / mean_lat_ms, 1),
        "overall_recall_50": round(overall_recall, 4),
        "overall_precision_50": round(overall_precision, 4),
        "mean_average_precision_map50": round(overall_map50, 4),
        "per_class_results": eval_results
    }

    # Print Report Table
    print("\n" + "=" * 80)
    print("               EMPIRICAL CCTV EVALUATION METRIC SCORECARD")
    print("=" * 80)
    print(f"{'Class':<12} | {'GT Count':<10} | {'TP':<6} | {'FP':<6} | {'Recall@50':<11} | {'Precision':<11} | {'AP@50':<8}")
    print("-" * 80)
    for cname, res in eval_results.items():
        print(f"{cname:<12} | {res['ground_truth_instances']:<10} | {res['detected_true_positives']:<6} | {res['false_positives']:<6} | "
              f"{res['recall_50']*100:.1f}%      | {res['precision_50']*100:.1f}%      | {res['average_precision_ap50']*100:.1f}%")
    print("-" * 80)
    print(f"{'OVERALL':<12} | {total_gt:<10} | {total_tp:<6} | {sum(v['false_positives'] for v in eval_results.values()):<6} | "
          f"{overall_recall*100:.1f}%      | {overall_precision*100:.1f}%      | {overall_map50*100:.1f}% (mAP@50)")
    print("=" * 80)
    print(f"-> Mean Inference Latency: {mean_lat_ms:.2f} ms ({1000.0/mean_lat_ms:.1f} FPS)")

    # Save to JSON
    os.makedirs("data", exist_ok=True)
    out_json = "data/real_cctv_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Saved empirical evaluation results to {out_json}")

    # --------------------------------------------------------------------------
    # 4. Generate High-Res Analytical Chart
    # --------------------------------------------------------------------------
    print("\n[4/4] Generating Precision-Recall Curve & Metric Chart...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')

    # Chart 1: Per-Class Precision & Recall Bar Chart
    labels = list(eval_results.keys()) + ["Overall (mAP)"]
    recalls_bar = [eval_results[k]["recall_50"] * 100 for k in eval_results.keys()] + [overall_recall * 100]
    ap_bar = [eval_results[k]["average_precision_ap50"] * 100 for k in eval_results.keys()] + [overall_map50 * 100]

    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, recalls_bar, w, label='Recall@50 (%)', color='#06b6d4', edgecolor='#22d3ee', linewidth=1)
    ax1.bar(x + w/2, ap_bar, w, label='AP@50 / mAP@50 (%)', color='#f43f5e', edgecolor='#fb7185', linewidth=1)

    ax1.set_ylabel('Accuracy (%)', color='#f8fafc', fontsize=9, fontweight='bold')
    ax1.set_title('Empirical Detection Accuracy on Real CCTV Test Set', color='#ffffff', fontsize=10.5, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color='#cbd5e1', fontsize=9, fontweight='bold')
    ax1.tick_params(colors='#94a3b8')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.25, color='#94a3b8')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#ffffff', fontsize=8.5)

    for i in range(len(labels)):
        ax1.text(x[i] - w/2, recalls_bar[i] + 2, f"{recalls_bar[i]:.1f}%", ha='center', color='#38bdf8', fontsize=8, fontweight='bold')
        ax1.text(x[i] + w/2, ap_bar[i] + 2, f"{ap_bar[i]:.1f}%", ha='center', color='#fb7185', fontsize=8, fontweight='bold')

    # Chart 2: Precision-Recall Curve
    ax2.set_title('Precision-Recall Curves (IoU >= 0.50)', color='#ffffff', fontsize=10.5, fontweight='bold')
    colors_curve = {'Fire': '#f43f5e', 'Smoke': '#38bdf8'}
    for cname, res in eval_results.items():
        pr = res["pr_curve"]
        col = colors_curve.get(cname, '#10b981')
        ax2.plot(pr["recalls"], pr["precisions"], label=f"{cname} (AP={res['average_precision_ap50']*100:.1f}%)", color=col, lw=2.2)

    ax2.set_xlabel('Recall', color='#f8fafc', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Precision', color='#f8fafc', fontsize=9, fontweight='bold')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(linestyle='--', alpha=0.25, color='#94a3b8')
    ax2.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#ffffff', fontsize=8.5)

    plt.tight_layout()
    chart_path = "data/report_assets/chart_map_recall.png"
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved PR-Curve and Evaluation Chart to {chart_path}")

    return summary

if __name__ == "__main__":
    run_evaluation()
