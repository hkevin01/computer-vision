#!/usr/bin/env bash
set -euo pipefail
echo "== Environment Diagnostics =="
echo "Hostname: $(hostname)"
echo "OS: $(uname -a)"

echo "\n-- GPU vendors/backends --"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA detected"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi
if command -v rocminfo >/dev/null 2>&1; then
  echo "ROCm/HIP detected"
  rocminfo | head -n 5 || true
fi

echo "\n-- OpenCV --"
python3 - <<'PY'
import cv2,sys
try:
    print('OpenCV version:', cv2.__version__)
    build = cv2.getBuildInformation()
    for line in build.splitlines():
        if 'CUDA' in line or 'OpenCL' in line:
            print(line)
except Exception as e:
    print('OpenCV python import failed:', e)
    sys.exit(0)
PY

echo "\n-- ONNX Runtime providers --"
python3 - <<'PY'
import onnxruntime as ort,sys
try:
    print('ONNXRuntime version:', ort.__version__)
    print('Available providers:', ort.get_all_providers())
except Exception as e:
    print('ONNXRuntime not available:', e)
    sys.exit(0)
PY

echo "\n-- TensorRT --"
if python3 - <<'PY'
try:
    import tensorrt as trt
    print(True)
except Exception:
    print(False)
PY
then
    python3 - <<'PY'
import tensorrt as trt
print('TensorRT version available')
PY
else
    echo "TensorRT not found or not importable"
fi

echo "\n-- Summary --"
echo "Diagnostic complete"
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
