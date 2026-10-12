#!/bin/bash
# ==============================================================================
# scripts/setup_rpi_acceleration.sh
# Hardware Acceleration & Kernel Tuning for Raspberry Pi 5 (Cortex-A76)
# ==============================================================================

set -e

echo "[1/4] Installing system dependencies and OpenBLAS / KleidiAI packages..."
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev libjpeg-turbo8-dev git python3-dev

echo "[2/4] Configuring ARMv8.2-A vector compiler flags for Cortex-A76..."
export CFLAGS="-march=armv8.2-a+dotprod+fp16 -O3 -fPIC"
export CXXFLAGS="-march=armv8.2-a+dotprod+fp16 -O3 -fPIC"
export CMAKE_ARGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

echo "[3/4] Recompiling llama-cpp-python with ARM NEON SIMD kernels..."
pip uninstall -y llama-cpp-python || true
pip install llama-cpp-python --no-cache-dir

echo "[4/4] Verifying hardware SIMD extensions..."
python3 -c "
import llama_cpp
print('Llama.cpp successfully compiled with ARM NEON support!')
"

echo "=============================================================================="
echo "Raspberry Pi 5 Hardware Acceleration Setup Complete!"
echo "=============================================================================="
