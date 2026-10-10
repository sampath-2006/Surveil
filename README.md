# 🛡️ Surveil: VLM-Based Danger Detection & 3D Mapping System

**Surveil** is an advanced edge-AI system specifically architected for the **Raspberry Pi 5 (16GB)**. It utilizes Vision Language Models (VLMs) to detect complex, contextual dangers (fire, smoke, conflict) and reconstructs dense metric 3D scene maps without requiring heavy GPU hardware.

Because large VLMs typically cause Out-of-Memory (OOM) crashes on embedded devices, this system implements a rigid **sequential model-loading architecture**, gated by a highly efficient real-time MobileNetV3 trigger.

---

## 🌟 Key Features

* **Real-time Trigger Gating:** Runs a continuous `MobileNetV3` ONNX loop at ~30 FPS to filter safe frames, saving heavy VLM compute for true anomalies.
* **VLM Danger Context:** Uses `Moondream 2B` (4-bit GGUF via llama.cpp) to contextually analyze the triggered anomaly and return JSON risk assessments.
* **VLM Open-Vocabulary Mapping:** Sequentially loads `Florence-2-Large` (Object Detection) and `DepthVLM-4B` (Metric Depth Estimation) every 5 minutes to feed into an Open3D reconstructor.
* **Memory & Thermal Aware:** Includes a `VLMManager` lock to guarantee only one large model is loaded at a time (peak memory ~4.4GB) and a `ThermalMonitor` that auto-throttles the Pi if temperatures exceed 80°C.
* **IoT Alerting:** Dispatches High/Critical alerts via an MQTT backend.

## ⚙️ Architecture

The system spins up 3 core Python threads:
1. **Surveillance Thread:** Handles `cv2` Capture -> SSIM frame skipping -> MobileNetV3 Trigger.
2. **Danger Analysis Thread (Async):** Sleeps until signaled by the trigger. Loads Moondream -> analyzes -> alerts via MQTT -> unloads.
3. **3D Mapping Thread (Scheduled):** Fires every 5 minutes (postponed if danger is active). Loads Florence-2 -> unloads -> loads DepthVLM -> unloads -> reconstructs map.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sampath-2006/Surveil.git
   cd Surveil
   ```

2. **Install dependencies:**
   *(Ensure you have Python 3.11+)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Hardware:**
   Edit `config/rpi5_16gb.yaml` to adjust camera endpoints, intervals, and thermal throttling limits.

## 🚀 Running the System

### Simulation Mode (No huge model weights required)
If you just cloned the repository, the VLM memory manager will automatically detect missing model weights (`.gguf` / `.onnx`) and fall back to **Simulation Mode**. 
You can run this immediately on your PC or Mac to watch the multi-threaded system interact, connect to your webcam, and dispatch mock alerts.

```bash
python src/main.py
```

### Production Mode (Downloading Models)
Before deploying on the Raspberry Pi 5, you must download the necessary INT4 and INT8 quantized models to an NVMe SSD.

```bash
python scripts/download_models.py
```
*(Note: Ensure you have an active cooler on the RPi 5. Sustained VLM inference will quickly thermal throttle the CPU without one!)*

---
*Built for the resource-constrained edge.*
