#!/usr/bin/env bash
set -euo pipefail

echo "== Environment Diagnosis =="

echo "* Host: $(uname -a)"

echo "\n-- GPU Vendors and Backends"
if command -v nvidia-smi &>/dev/null; then
  echo "NVIDIA GPU detected"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  if command -v nvcc &>/dev/null; then
    nvcc --version | sed -n '1,3p'
  fi
else
  echo "No NVIDIA detected (nvidia-smi missing)"
fi

if command -v rocminfo &>/dev/null; then
  echo "ROCm detected"
  rocminfo | head -n 20 || true
else
  echo "No ROCm detected"
fi

echo "\n-- OpenCV"
python3 - <<'PY'
import cv2
print('OpenCV version:', cv2.__version__)
build_info = cv2.getBuildInformation()
print('OpenCV build info excerpt:')
for line in build_info.splitlines():
    if any(k in line for k in ('CUDA', 'ONNX', 'TBB', 'OpenCL', 'GStreamer')):
        print(' ', line)
PY

echo "\n-- ONNX Runtime providers"
python3 - <<'PY'
import sys
try:
    import onnxruntime as ort
    print('ONNX Runtime version:', ort.__version__)
    print('Providers:', ort.get_available_providers())
except Exception as e:
    print('ONNX Runtime not available:', e)
    sys.exit(0)
PY

echo "\n-- TensorRT"
if command -v trtexec &>/dev/null; then
  trtexec --version || true
else
  echo "TensorRT (trtexec) not found"
fi

echo "\nDiagnosis complete."
