# Model Improvements & Performance Evaluation Framework

This document outlines the evaluation metrics, benchmarking standards, and targeted architectural improvements for the multi-modal VLM edge surveillance and 3D mapping pipeline.

---

## 1. Performance Evaluation Metrics

To rigorously assess whether the models meet production safety and efficiency standards on edge hardware (Raspberry Pi 5 16GB), performance is evaluated across three core pillars:

### A. Danger Detection & VLM Quality (Moondream 2B & MobileNetV3)

| Metric | Target | Description & Formula |
| :--- | :--- | :--- |
| **Hazard Recall / Sensitivity** | **> 95.0%** | $\frac{TP}{TP + FN}$ on critical hazard scenarios (fire, smoke, weapons, physical assault). In security surveillance, false negatives can be catastrophic. |
| **False Alarm Rate (FAR)** | **< 1 / 24h** | Rate of false positives during continuous operation (e.g., mistaking cigarette smoke, steam, or red reflections for fire). |
| **F1-Score per Hazard Class** | **> 0.88** | Harmonic mean of Precision and Recall across individual categories: `FIRE`, `SMOKE`, `CONFLICT`, `INTRUSION`. |
| **JSON Schema Compliance** | **> 99.5%** | Percentage of VLM inference responses that strictly parse into the expected JSON structure: `{"type": "...", "location": "...", "severity": "...", "action": "..."}`. |
| **Time-to-First-Token (TTFT)** | **< 1.5s** | Latency from trigger signal arrival until the first token is emitted by the VLM on edge CPU. |
| **End-to-End Alert Latency** | **< 3.5s** | Total elapsed time from physical hazard occurrence in frame to MQTT alert broadcast and UI alert notification. |

### B. 3D Spatial Reconstruction & Depth Metrics (Mapping Pipeline)

| Metric | Target | Description & Formula |
| :--- | :--- | :--- |
| **AbsRel (Absolute Relative Error)** | **< 0.12** | Metric depth accuracy comparing predicted pixel depth $d$ to ground truth $\hat{d}$: $\frac{1}{N} \sum \frac{\|d - \hat{d}\|}{d}$. |
| **RMSE (Root Mean Square Error)** | **< 0.35 m** | Standard deviation of spatial depth errors in indoor environments: $\sqrt{\frac{1}{N}\sum (d - \hat{d})^2}$. |
| **Threshold Accuracy ($\delta < 1.25$)** | **> 85.0%** | Percentage of pixels where $\max(\frac{d}{\hat{d}}, \frac{\hat{d}}{d}) < 1.25$. |
| **mAP@50 (Room Object Detection)** | **> 0.70** | Mean Average Precision at IoU 0.50 for key spatial objects (doors, furniture, entry points, obstructions). |
| **Novel View PSNR & SSIM** | **PSNR > 24 dB, SSIM > 0.82** | Visual quality and structural fidelity of rasterized 2D synthetic frames generated from the 3D map if the camera feed is interrupted. |

### C. Edge Hardware Efficiency & Resource Constraints (Raspberry Pi 5)

| Metric | Target | Description |
| :--- | :--- | :--- |
| **Peak RAM Footprint (RSS)** | **< 6.5 GB** | Total resident system memory during peak VLM loading and inference, leaving plenty of headroom within 16GB. |
| **Trigger Inference Latency** | **< 20 ms** | MobileNetV3 evaluation time per keyframe (sustaining 50+ FPS processing capability). |
| **Thermal Operating Envelope** | **< 70°C** | Package temperature maintained with active cooling; zero throttle events triggered at 80°C threshold. |
| **Model Swap Overhead** | **< 1.2s** | Latency to unload one VLM model and memory-map another using `VLMManager` on NVMe storage. |

---

## 2. Model Improvement Roadmap

The following prioritized roadmap details concrete model-level improvements:

### Improvement 1: Path Resolution & Physical Model Loading
* **Status**: Immediate Priority.
* **Objective**: Configure absolute and relative model path resolution across `config/rpi5_16gb.yaml`, `vlm_manager.py`, and `trigger.py` so physical weights (`models/moondream2-q4_k_m.gguf`, `models/mobilenetv3_small_int8.onnx`) load directly without tripping mock fallbacks.

### Improvement 2: Constrained Grammar Decoding (GBNF Schema)
* **Objective**: Eliminate JSON parsing errors and hallucinations.
* **Implementation**:
  * Integrate `llama_cpp.LlamaGrammar` using a strict GBNF grammar definition corresponding to the required alert schema.
  * Forces the model's logits sampler to only pick valid JSON tokens, guaranteeing 100% parseable output with zero retry overhead.

### Improvement 3: Real-Time Edge Depth Upgrade (Depth-Anything-V2 ONNX)
* **Objective**: Replace DepthVLM proxy with a real, high-speed metric depth engine.
* **Implementation**:
  * Integrate **Depth-Anything-V2-Small** quantized to INT8 ONNX (~25 MB).
  * Runs at ~60–90ms on Raspberry Pi 5 CPU without requiring multi-gigabyte LLM backbones.
  * Direct dense depth map feeding into `MapReconstructor` for true 3D point cloud generation.

### Improvement 4: Domain-Specific QLoRA Fine-Tuning
* **Objective**: Sharpen Moondream 2B accuracy on real-world CCTV hazard datasets.
* **Datasets**:
  * **D-Fire / Smoke-100k**: Specialized fire and industrial smoke detection in varying lighting conditions.
  * **UCF-Crime & RWF-2000**: Physical violence, weapons, and intrusion scenarios.
* **Deployment**:
  * Train with 4-bit QLoRA on a cloud/GPU workstation.
  * Merge adapters and export to GGUF format with `q4_k_m` or `q5_k_m` quantization for edge deployment.

### Improvement 5: Hardware Acceleration (ARM NEON / NPU Integration)
* **Objective**: Maximize throughput on edge hardware.
* **Implementation**:
  * Compile `llama-cpp-python` with `GGML_CPU_ARM_ARCH=armv8.2-a` and ARM NEON / KleidiAI kernels for the Cortex-A76 processor.
  * Offload MobileNetV3 and YOLO trigger models to dedicated accelerator hardware (e.g., Raspberry Pi AI HAT / Hailo-8 M.2 module at 13–26 TOPS).

### Improvement 6: Automated Evaluation Suite (`scripts/benchmark_models.py`)
* **Objective**: Provide automated scorecard validation.
* **Implementation**:
  * A test script that passes sample danger/safe images through both the trigger and Moondream VLM.
  * Automatically calculates and logs TTFT, total latency, peak RAM delta, and detection confidence.
