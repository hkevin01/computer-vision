#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

echo "Project root: $ROOT_DIR"

DETECTED_CUDA=0
DETECTED_ROCM=0

if command -v nvcc &>/dev/null || command -v nvidia-smi &>/dev/null; then
  DETECTED_CUDA=1
fi
if command -v rocminfo &>/dev/null; then
  DETECTED_ROCM=1
fi

USE_CUDA=OFF
USE_HIP=OFF
WITH_ONNX=ON
WITH_TENSORRT=OFF

if [ "$DETECTED_CUDA" -eq 1 ] && [ "$DETECTED_ROCM" -eq 0 ]; then
  echo "Detected NVIDIA/CUDA; enabling CUDA path"
  USE_CUDA=ON
  WITH_TENSORRT=ON
fi
if [ "$DETECTED_ROCM" -eq 1 ] && [ "$DETECTED_CUDA" -eq 0 ]; then
  echo "Detected AMD/ROCm; enabling HIP path"
  USE_HIP=ON
fi

echo "Build configuration: USE_CUDA=$USE_CUDA USE_HIP=$USE_HIP WITH_ONNX=$WITH_ONNX WITH_TENSORRT=$WITH_TENSORRT"

cmake -B "$BUILD_DIR" -S "$ROOT_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUSE_CUDA=${USE_CUDA} \
  -DUSE_HIP=${USE_HIP} \
  -DWITH_ONNX=${WITH_ONNX} \
  -DWITH_TENSORRT=${WITH_TENSORRT} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "$BUILD_DIR" --parallel "$(nproc)"

echo "Build complete. Artifacts in: $BUILD_DIR"
