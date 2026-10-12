# 🛡️ Surveil: Edge-Optimized Multi-Modal VLM Danger Detection & 3D Mapping System

[![Hardware](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20(16GB)-C51A4A?logo=raspberry-pi)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Inference Engine](https://img.shields.io/badge/Inference-llama.cpp%20%7C%20ONNX%20Runtime-blue)](https://github.com/ggerganov/llama.cpp)
[![Frontend](https://img.shields.io/badge/Dashboard-React%2018%20%2B%20Vite-61DAFB?logo=react)](https://vitejs.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Surveil** is an edge-native, multi-modal Vision Language Model (VLM) surveillance and 3D spatial room mapping system built specifically for the **Raspberry Pi 5 (16GB)**. It provides real-time danger classification (fire, smoke, physical conflicts, intrusion) and dense 3D room reconstruction with on-demand **Novel View Synthesis** for camera loss recovery—all running locally on resource-constrained edge hardware without external cloud dependencies.

---

## 🌟 Key Capabilities

* **Real-Time Trigger Gating (30 FPS)**: An INT8-quantized `MobileNetV3` binary trigger evaluates keyframes gated by SSIM motion variance to filter safe frames, preserving heavy VLM compute for true anomalies.
* **Contextual VLM Danger Analysis**: High-severity events invoke `Moondream 2B` (4-bit GGUF via `llama-cpp-python`) to parse visual hazards and output structured JSON risk assessments (`{type, location, severity, action}`).
* **Dense 3D Room Reconstruction**: Fuses open-vocabulary object detections and depth maps into standard **Stanford Polygon (`.ply`) 3D Point Clouds** with 3D bounding coordinates and spatial dimensions.
* **Novel View Synthesis (Scene Recovery)**: Rasterizes virtual 2D camera angles (Isometric, Perspective Center, Top-Down Ortho) from the active 3D map to maintain situational awareness if physical video feed drops.
* **Live MJPEG Surveillance Feed**: Streams live video at 30 FPS (`/api/feed`) directly to the web dashboard with synthetic live-feed fallback if physical camera access is restricted.
* **RPi Edge Production Hardening**:
  * **Memory Manager (`VLMManager`)**: Enforces single-model sequential swapping with explicit garbage collection to prevent Out-of-Memory (OOM) crashes.
  * **Thermal Protection (`ThermalMonitor`)**: Actively throttles non-critical cycles if CPU temperatures approach 80°C.
  * **Auto-Boot Watchdog (`systemd`)**: Systemd watchdog service for automatic startup and crash recovery.
  * **NVMe Storage Pruning (`DataPruner`)**: Daily daemon pruning maps and logs older than 7 days.
  * **Secured REST API**: Token-authenticated endpoints (`X-API-Key`) for telemetry and control.
* **Interactive React Dashboard**: Sleek dark-mode interface exposing live video feeds, 3D point cloud metadata, novel view rendering, and real-time system state.

---

## 🏗️ Multi-Threaded Architecture

Surveil coordinates 5 synchronized background threads:

```
                  +----------------------------------------------+
                  |         Camera Capture (FrameEngine)         |
                  +----------------------------------------------+
                                         |
                                (SSIM Keyframing)
                                         v
   +--------------------------------------------------------------------------+
   | Thread 1: Surveillance Loop (Continuous)                                 |
   |   - Evaluates frame through MobileNetV3 (INT8 ONNX)                      |
   |   - Triggers anomaly signal if confidence > threshold                    |
   +--------------------------------------------------------------------------+
                         |                                  |
               (Hazard Detected)                 (Periodic Timer: 300s)
                         v                                  v
   +---------------------------------------+  +-------------------------------+
   | Thread 2: Danger Analysis (Async)     |  | Thread 3: 3D Mapping (Sched)  |
   |   - Loads Moondream 2B (GGUF)         |  |   - Loads Florence-2 ONNX     |
   |   - Strict JSON hazard classification |  |   - Loads DepthVLM / Depth-V2 |
   |   - Dispatches IoT MQTT alerts        |  |   - Exports Stanford .ply Map |
   |   - Memory unloads to free RAM        |  |   - Synthesizes 2D Snapshots  |
   +---------------------------------------+  +-------------------------------+
                         |                                  |
                         +-----------------+----------------+
                                           v
   +--------------------------------------------------------------------------+
   | Thread 4: NVMe Data Pruning Daemon (7-day TTL storage guard)              |
   | Thread 5: Flask REST API Server (Port 5000)                              |
   +--------------------------------------------------------------------------+
```

---

## 📁 3D Map Format Specifications

Reconstructed spatial scenes are saved to `data/maps/`:
* **File Format**: Standard **Stanford Polygon (`.ply`) Point Cloud (ASCII 1.0)** containing metric XYZ point coordinates and RGB vertex color channels.
* **Spatial Metadata (`map_v{N}_metadata.json`)**: Contains metric bounding dimensions, total vertex count, 3D bounding boxes for detected objects, and camera intrinsics.
* **Synthesized 2D Snapshots (`data/maps/snapshots/`)**:
  * `map_v{N}_isometric_projection.png`
  * `map_v{N}_perspective_center.png`
  * `map_v{N}_top_down_ortho.png`

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
* Python 3.10+ (Conda or virtualenv recommended)
* Node.js 18+ & npm
* C++ Build Tools (for native `llama-cpp-python` compilation)

### 2. Clone and Setup
```bash
git clone https://github.com/sampath-2006/Surveil.git
cd Surveil

# Install Python backend dependencies
pip install -r requirements.txt

# Install frontend dashboard dependencies
cd frontend
npm install
cd ..
```

### 3. Download Model Weights
Download the quantized INT8 and GGUF model weights into `models/`:
```bash
python scripts/download_models.py
```
*(If models are not downloaded, the system gracefully falls back to high-fidelity simulation mode for local testing).*

---

## 💻 Running the System

### Run the Backend (Python)
```bash
python src/main.py
```
* Backend API runs on: `http://127.0.0.1:5000`

### Run the Dashboard (React / Vite)
In a separate terminal:
```bash
cd frontend
npm run dev
```
* Open your browser at: **`http://localhost:5173`**

---

## 📡 REST API Reference

All protected endpoints require the `X-API-Key` header (default: `dev_secret_key`).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/state` | Returns thread loop statuses, active VLM, and latest 3D map metadata. |
| `GET` | `/api/feed` | Real-time MJPEG live camera stream (embeddable via `<img>`). |
| `GET` | `/api/current_frame` | Single latest JPEG snapshot from surveillance feed. |
| `GET` | `/api/map_info` | Returns Stanford PLY path, spatial bounding extents, and detected objects. |
| `GET` | `/api/map_snapshot?view={name}` | Serves rendered 2D synthesized perspectives (`isometric_projection`, `perspective_center`, `top_down_ortho`). |
| `POST` | `/api/render_map_view` | On-demand Novel View Synthesis: rasterizes a new 2D snapshot from the 3D map. |
| `POST` | `/api/trigger_danger` | Manually injects an anomaly event to test Moondream 2B danger analysis. |
| `POST` | `/api/trigger_mapping` | Manually forces an immediate 3D room reconstruction cycle. |

---

## 📊 Evaluation Metrics & Improvement Roadmap

A comprehensive benchmark framework is documented in [model_improvements.md](model_improvements.md), including:
* **Danger Detection Metrics**: Recall (>95%), False Alarm Rate (<1/24h), F1-Score (>0.88), TTFT (<1.5s), End-to-End Latency (<3.5s).
* **3D Mapping Metrics**: Absolute Relative Depth Error (AbsRel < 0.12), RMSE (<0.35m), PSNR/SSIM (>24 dB / >0.82).
* **Edge Hardware Constraints**: Peak RAM Footprint (<6.5 GB on 16GB budget), Trigger Latency (<20 ms).

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
