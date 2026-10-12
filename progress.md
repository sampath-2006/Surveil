# Project Progress Log

All updates and implementation steps are tracked here.

## 2026-10-10
* **11:34 AM** - Finalized the System Design Plan (v3) tailored for the Raspberry Pi 5 16GB. Saved as `design_plan.md`.
* **11:34 AM** - Created the project directory structure (`src`, `config`, `tests`, `data`, `scripts`, `models`, `docker`, `systemd`).
* **11:34 AM** - Generated core project files: `requirements.txt` and `pyproject.toml`.
* **11:34 AM** - Created `config/rpi5_16gb.yaml` with optimized hardware/timing settings.
* **11:34 AM** - Implemented the main entry point `src/main.py` scaffolding the 3 core multi-threaded loops (Surveillance, Danger Analysis, Mapping).
* **11:36 AM** - Initialized tracking in `progress.md` and created the initial git repository.
* **11:42 AM** - Implemented structured logging in `src/utils/logger.py`.
* **11:42 AM** - Implemented configuration loader in `src/core/config.py`.
* **11:42 AM** - Developed SSIM-based keyframe selection in `src/capture/keyframe_selector.py`.
* **11:42 AM** - Built the `FrameEngine` in `src/capture/frame_engine.py` for camera integration.
* **11:43 AM** - Updated `src/main.py` to integrate threading, config loading, and the `FrameEngine`.
* **11:46 AM** - Implemented `VLMManager` in `src/optimization/vlm_manager.py` to handle sequential VLM loading and prevent OOM errors on RPi.
* **11:46 AM** - Implemented `ThermalMonitor` in `src/optimization/thermal_monitor.py` to prevent CPU throttling.
* **11:48 AM** - Developed `DangerTrigger` (`src/detection/trigger.py`) for lightweight MobileNetV3 gating.
* **11:48 AM** - Developed `VLMDangerAnalyzer` (`src/detection/vlm_danger.py`) as a Moondream 2B inference wrapper.
* **11:48 AM** - Integrated the full danger detection pipeline into the `main.py` multi-threading architecture.
* **11:51 AM** - Developed `VLMObjectDetector` (`src/mapping/vlm_object_detector.py`) for Florence-2 open-vocabulary object detection.
* **11:51 AM** - Developed `VLMDepthEstimator` (`src/mapping/vlm_depth_estimator.py`) for DepthVLM-4B metric depth prediction.
* **11:51 AM** - Created `MapReconstructor` (`src/mapping/reconstructor.py`) to handle 3D point cloud generation.
* **11:51 AM** - Fully wired the scheduled 3D Mapping thread in `main.py`, orchestrating sequential VLM loading via the `VLMManager`.
* **11:55 AM** - Implemented `AlertManager` (`src/alerting/alert_manager.py`) and `MQTTAlertBackend` (`src/alerting/backends/mqtt_backend.py`) for IoT-native danger notifications, wiring it into the Danger Analysis loop.
* **11:57 AM** - Completed the codebase by injecting the real `llama-cpp-python` and `onnxruntime` inference calls into the VLM Wrappers (`vlm_danger.py`, `vlm_object_detector.py`, `vlm_depth_estimator.py`, `trigger.py`) with fallback "mock" modes so the system can run and be tested immediately before massive model weights finish downloading.
* **11:57 AM** - Created `scripts/download_models.py` to automate downloading the ONNX and GGUF VLM weights to the NVMe disk.
* **12:11 PM** - Developed a fully integrated React/Vite Dashboard in `frontend/` to visualize the system state, connected to the Python backend via a newly implemented Flask API.
* **12:27 PM** - Upgraded the `MapReconstructor` to include Novel View Synthesis, automatically rasterizing and exporting virtual 2D `.png` snapshots derived from the constructed 3D map geometry.
* **12:35 PM** - Transitioned the system to a production-ready edge state by implementing three critical features:
  1. **Data Pruning (`src/core/pruner.py`)**: A daily cron thread that deletes maps/logs older than 7 days to protect the NVMe drive.
  2. **Auto-Boot Configuration (`systemd/vlm-surveil.service`)**: Systemd service to auto-start the Python backend on Raspberry Pi boot and auto-restart on crash.
  3. **API Security**: Added `X-API-Key` authentication to `src/api.py` and synced the React frontend to pass secure headers to prevent unauthorized triggers.
* **1:30 PM** - Successfully implemented secure HuggingFace authentication to bypass 401/404 errors and completely downloaded the physical Moondream2 (`.gguf`), Florence-2 (`.onnx`), and MobileNetV3 (`.onnx`) models into the `models/` directory for physical hardware deployment.
* **10:00 PM** - Compiled and installed the hardware-accelerated AI inference engines (`llama-cpp-python` and `onnxruntime`). The backend `VLMManager` is now fully operational in production mode, loading physical models directly into RAM instead of using the Mock simulator!

## 2026-10-12
* **10:35 AM** - Formulated and documented the Model Improvements & Performance Evaluation Framework in `model_improvements.md`, defining quantitative targets for hazard recall (>95%), false alarm rate (<1/24h), latency (<3.5s), depth AbsRel (<0.12), and edge memory constraints.
* **10:38 AM** - Upgraded `VLMManager` and `DangerTrigger` with resilient `models/` directory resolution and enforced `float32` tensor casting for MobileNetV3 ONNX inference. Launched both the Python backend and React Vite dashboard (`http://localhost:5173/`) in live operation.
* **10:50 AM** - Implemented live MJPEG surveillance streaming (`/api/feed`), integrated Stanford PLY 3D Point Cloud storage & metadata inspector (`data/maps/map_v1.ply`), and added on-demand Novel View Synthesis controls (`POST /api/render_map_view`) with multi-perspective switching (Isometric, Perspective Center, Top-Down Ortho).
* **10:55 AM** - Revamped `README.md` with full architectural diagrams, 3D point cloud format specs, API reference table, and edge deployment guides. Successfully committed and pushed clean codebase to GitHub repository (`origin/main`).
