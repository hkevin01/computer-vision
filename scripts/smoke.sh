#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
REPORTS_DIR="$ROOT_DIR/reports/smoke"
mkdir -p "$REPORTS_DIR"

echo "Running smoke tests..."

run_test() {
  bin="$1"
  out="$REPORTS_DIR/$(basename "$bin").log"
  if [ -x "$bin" ]; then
    echo "Running $bin"
    "$bin" > "$out" 2>&1 || { echo "$bin failed, see $out"; return 2; }
  else
    echo "Missing binary: $bin"; return 1
  fi
}

# Expected test binaries
tests=(
  "$BUILD_DIR/test_programs/test_onnx_load"
  "$BUILD_DIR/test_programs/test_stereo_cpu"
  "$BUILD_DIR/test_programs/test_pointcloud"
  "$BUILD_DIR/test_programs/test_camera_list"
)

failed=0
for t in "${tests[@]}"; do
  run_test "$t" || failed=$((failed+1))
done

if [ "$failed" -ne 0 ]; then
  echo "Smoke tests failed: $failed failures"
  exit 1
fi

echo "Smoke tests passed. Logs in $REPORTS_DIR"
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
REPORT_DIR="$ROOT_DIR/reports/smoke"
mkdir -p "$REPORT_DIR"

echo "Running smoke tests (headless). Reports -> $REPORT_DIR"

echo "1/4: test_onnx_load"
if [ -x "$BUILD_DIR/test_programs/test_onnx_load" ]; then
  "$BUILD_DIR/test_programs/test_onnx_load" --model "$ROOT_DIR/models/hitnet.onnx" > "$REPORT_DIR/test_onnx_load.log" 2>&1 || exit 2
else
  echo "test_onnx_load binary not found; skipping" > "$REPORT_DIR/test_onnx_load.log"
fi

echo "2/4: test_stereo_cpu"
if [ -x "$BUILD_DIR/test_programs/test_stereo_cpu" ]; then
  "$BUILD_DIR/test_programs/test_stereo_cpu" --left "$ROOT_DIR/data/stereo_images/left.png" --right "$ROOT_DIR/data/stereo_images/right.png" --out "$REPORT_DIR/disparity.png" > "$REPORT_DIR/test_stereo_cpu.log" 2>&1 || exit 3
else
  echo "test_stereo_cpu binary not found; skipping" > "$REPORT_DIR/test_stereo_cpu.log"
fi

echo "3/4: test_pointcloud"
if [ -x "$BUILD_DIR/test_programs/test_pointcloud" ]; then
  "$BUILD_DIR/test_programs/test_pointcloud" --disparity "$REPORT_DIR/disparity.png" --out "$REPORT_DIR/cloud.ply" > "$REPORT_DIR/test_pointcloud.log" 2>&1 || exit 4
else
  echo "test_pointcloud binary not found; skipping" > "$REPORT_DIR/test_pointcloud.log"
fi

echo "4/4: test_camera_list"
if [ -x "$BUILD_DIR/test_programs/test_camera_list" ]; then
  "$BUILD_DIR/test_programs/test_camera_list" > "$REPORT_DIR/test_camera_list.log" 2>&1 || exit 5
else
  echo "test_camera_list binary not found; skipping" > "$REPORT_DIR/test_camera_list.log"
fi

echo "Smoke run complete. Logs in $REPORT_DIR"
