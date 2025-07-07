#!/bin/bash
set -e

echo "Building Stereo Vision Project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable: ./stereo_vision_app"
echo "Test executable: ./test_stereo_vision"
