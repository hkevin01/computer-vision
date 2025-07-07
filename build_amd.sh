#!/bin/bash
# Build script optimized for AMD GPUs using HIP

set -e

mkdir -p build_amd
cd build_amd

echo "Configuring for AMD GPU with HIP support..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_HIP=ON \
    -DUSE_CUDA=OFF \
    -DCMAKE_PREFIX_PATH="/opt/rocm" \
    -DHIP_ROOT_DIR="/opt/rocm" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Building with HIP support..."
make -j$(nproc)

echo "Build completed! Executable: ./stereo_vision_app"
