# Quick Start Guide

This guide will help you get the Computer Vision Stereo Processing Library up and running in just a few minutes.

## Prerequisites

Before installing the library, ensure you have the following dependencies:

### System Requirements

=== "Ubuntu 20.04/22.04"

    ```bash
    # Update package list
    sudo apt update

    # Install build essentials
    sudo apt install -y build-essential cmake git pkg-config

    # Install OpenCV
    sudo apt install -y libopencv-dev libopencv-contrib-dev

    # Install PCL (Point Cloud Library)
    sudo apt install -y libpcl-dev

    # Install Qt5 for GUI components
    sudo apt install -y qt5-default qttools5-dev-tools

    # Install additional dependencies
    sudo apt install -y libeigen3-dev libceres-dev
    ```

=== "CentOS/RHEL 8+"

    ```bash
    # Install development tools
    sudo dnf groupinstall "Development Tools"
    sudo dnf install cmake git pkg-config

    # Install OpenCV (may need EPEL repository)
    sudo dnf install opencv-devel

    # Install Qt5
    sudo dnf install qt5-qtbase-devel qt5-qttools-devel

    # Install PCL
    sudo dnf install pcl-devel
    ```

=== "macOS"

    ```bash
    # Install Homebrew if not already installed
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Install dependencies
    brew install cmake opencv pcl qt5 eigen ceres-solver
    ```

### GPU Support (Optional)

For GPU acceleration, install CUDA or HIP:

=== "NVIDIA CUDA"

    ```bash
    # Download and install CUDA toolkit from NVIDIA
    # https://developer.nvidia.com/cuda-downloads

    # Verify installation
    nvcc --version
    nvidia-smi
    ```

=== "AMD HIP"

    ```bash
    # Install ROCm/HIP (Ubuntu)
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt update
    sudo apt install rocm-dev hip-dev
    ```

## Installation Methods

### Method 1: Quick Build Script

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/computer-vision-project/computer-vision.git
cd computer-vision

# Run the setup script
./scripts/setup_dev_environment.sh

# Build the project
./build.sh
```

### Method 2: Manual CMake Build

For more control over the build process:

```bash
# Clone and navigate to project
git clone https://github.com/computer-vision-project/computer-vision.git
cd computer-vision

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_CUDA=ON \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_TESTS=ON \
      ..

# Build (use all available cores)
make -j$(nproc)

# Optional: Install system-wide
sudo make install
```

### Method 3: Docker Installation

For a containerized environment:

```bash
# Pull the pre-built image
docker pull computer-vision/stereo-processing:latest

# Or build from source
git clone https://github.com/computer-vision-project/computer-vision.git
cd computer-vision
docker build -t computer-vision-local .

# Run with camera access
docker run -it --rm \
  --device=/dev/video0:/dev/video0 \
  -v $(pwd)/data:/app/data \
  computer-vision/stereo-processing:latest
```

## Build Configuration Options

The build system supports various configuration options:

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: Debug, Release, RelWithDebInfo |
| `ENABLE_CUDA` | `AUTO` | Enable CUDA GPU acceleration |
| `ENABLE_HIP` | `AUTO` | Enable AMD HIP GPU acceleration |
| `BUILD_EXAMPLES` | `ON` | Build example applications |
| `BUILD_TESTS` | `ON` | Build unit tests |
| `BUILD_GUI` | `ON` | Build Qt-based GUI applications |
| `ENABLE_ONNX` | `OFF` | Enable ONNX runtime for AI models |
| `USE_SYSTEM_OPENCV` | `ON` | Use system OpenCV instead of bundled |

Example custom configuration:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DENABLE_CUDA=OFF \
      -DBUILD_GUI=OFF \
      -DENABLE_ONNX=ON \
      ..
```

## Verification

After building, verify the installation:

### 1. Run Tests

```bash
# In build directory
make test

# Or with CTest for detailed output
ctest --output-on-failure
```

### 2. Try Example Applications

```bash
# Basic stereo processing
./build/stereo_vision_app_simple --left data/sample_images/left.jpg --right data/sample_images/right.jpg

# Live camera processing (requires connected stereo cameras)
./build/stereo_vision_app --camera-index 0

# Calibration tool
./build/live_stereo_tuning
```

### 3. Check Library Installation

Create a simple test program:

```cpp
// test_installation.cpp
#include "stereo_vision_core.hpp"
#include <iostream>

int main() {
    stereo_vision::StereoProcessor processor;
    std::cout << "Library loaded successfully!" << std::endl;
    std::cout << "OpenCV version: " << cv::getVersionString() << std::endl;
    return 0;
}
```

Compile and run:

```bash
g++ -std=c++17 test_installation.cpp -lopencv_core -lopencv_imgproc -lstereo_vision_core -o test_installation
./test_installation
```

## Troubleshooting

### Common Issues

#### OpenCV Not Found

```bash
# Install OpenCV development headers
sudo apt install libopencv-dev

# Or specify OpenCV path manually
cmake -DOpenCV_DIR=/path/to/opencv/build ..
```

#### Qt5 Issues

```bash
# Install Qt5 development packages
sudo apt install qt5-default qttools5-dev-tools

# For newer Ubuntu versions (22.04+)
sudo apt install qtbase5-dev qttools5-dev-tools
```

#### CUDA Compilation Errors

```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Check CUDA compatibility
nvcc --version
```

#### Permission Issues

```bash
# For camera access
sudo usermod -a -G video $USER
# Log out and log back in

# For device access in Docker
docker run --privileged ...
```

### Performance Issues

If you experience slow performance:

1. **Enable GPU acceleration** (CUDA/HIP)
2. **Use Release build** (`-DCMAKE_BUILD_TYPE=Release`)
3. **Optimize for your CPU** (`-DCMAKE_CXX_FLAGS="-march=native"`)
4. **Adjust thread count** in configuration files

### Getting Help

If you're still having issues:

1. Check our [FAQ](../user-guide/faq.md)
2. Search [GitHub Issues](https://github.com/computer-vision-project/computer-vision/issues)
3. Create a new issue with:
   - Your operating system and version
   - Build configuration used
   - Complete error messages
   - Steps to reproduce

## Next Steps

Now that you have the library installed:

1. **Try the tutorials**: Start with [Basic Stereo Processing](../tutorials/basic-stereo.md)
2. **Explore examples**: Check out the `examples/` directory
3. **Read the user guide**: Learn about [core concepts](../user-guide/overview.md)
4. **Configure your setup**: Optimize [streaming performance](../user-guide/streaming.md)

---

!!! success "Installation Complete!"
    You're ready to start processing stereo images! Try running the example applications or jump into our tutorials.

!!! tip "Performance Tip"
    For the best performance, make sure to build in Release mode and enable GPU acceleration if you have a compatible graphics card.
