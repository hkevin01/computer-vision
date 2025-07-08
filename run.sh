#!/usr/bin/env bash
# run.sh - Build and run the Stereo Vision 3D Point Cloud application
#
# Usage:
#   ./run.sh [--console] [--left <left_img_dir>] [--right <right_img_dir>] [--other-args]
#
# This script auto-detects the platform and builds the project if needed.
# It then runs the main application with any arguments you provide.

set -e

# Detect platform
OS="$(uname -s)"

# Set build script and binary paths
if [[ "$OS" == "Linux" ]]; then
    if lspci | grep -i amd &>/dev/null && command -v hipcc &>/dev/null; then
        BUILD_SCRIPT="./build_amd.sh"
        BINARY="./build_amd/stereo_vision_app"
    elif lspci | grep -i nvidia &>/dev/null && command -v nvcc &>/dev/null; then
        BUILD_SCRIPT="./build.sh"
        BINARY="./build/stereo_vision_app"
    else
        BUILD_SCRIPT="./build.sh"
        BINARY="./build/stereo_vision_app"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    BUILD_SCRIPT="./build.sh"
    BINARY="./build/stereo_vision_app"
elif [[ "$OS" =~ MINGW|MSYS|CYGWIN ]]; then
    BUILD_SCRIPT="N/A"
    BINARY="./build/Release/stereo_vision_app.exe"
    echo "Please build the project using CMake/Visual Studio on Windows."
    echo "Then run: $BINARY [args]"
    exit 1
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Build if binary does not exist
if [[ ! -f "$BINARY" ]]; then
    echo "[run.sh] Building project using $BUILD_SCRIPT ..."
    $BUILD_SCRIPT
fi

# Run the application
if [[ -f "$BINARY" ]]; then
    echo "[run.sh] Running: $BINARY $@"
    $BINARY "$@"
else
    echo "[run.sh] Error: Could not find built binary: $BINARY"
    exit 2
fi 