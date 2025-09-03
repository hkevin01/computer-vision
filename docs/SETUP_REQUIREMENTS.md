# Setup Requirements (Ubuntu 22.04 / 24.04)

Minimum known-good versions

- OpenCV: >= 4.5 (4.8+ recommended for ONNX/TensorRT)
- PCL: >= 1.12
- VTK: >= 9.0
- Qt: >= 6.2
- ONNX Runtime: >= 1.15
- TensorRT: >= 8.5 (optional)
- CUDA: >= 11.0 (if using NVIDIA)
- ROCm: >= 5.0 (if using AMD)

Install notes

- Use Ubuntu packages where possible for PCL/VTK. For OpenCV, prefer building from source when using CUDA/ONNX integration.
- ONNX Runtime should be installed with CPU-only pip wheel for CI:

```bash
python3 -m pip install onnxruntime
```

For CUDA-enabled ONNX Runtime on Ubuntu, install the matching wheel from the ONNX Runtime releases.
Ubuntu supported versions: 22.04, 24.04

Recommended minimum dependency versions (known-good)

- OpenCV: >= 4.5 (4.8+ recommended for ONNX/TensorRT)
- PCL: >= 1.12
- VTK: >= 9.0
- Qt6: >= 6.2
- ONNX Runtime: >= 1.15
- TensorRT: >= 8.5 (optional)
- CUDA: >= 11.0 (if using NVIDIA)
- ROCm: >= 5.0 (if using AMD)

Install notes

- Prefer system packages for Qt6 and OpenCV when available; for ONNX Runtime and TensorRT, use provider-specific installs.
- On Ubuntu, install build-time deps: git, build-essential, cmake, pkg-config, libopencv-dev, libpcl-dev, libvtk-dev, qt6-base-dev, python3, python3-pip

Common pitfalls

- PCL and VTK mismatched versions can break point cloud IO.
- ONNX Runtime without GPU providers will fall back to CPU; ensure `onnxruntime-gpu` or `onnxruntime-directml` are installed if needed.
- TensorRT requires compatible CUDA and driver versions.

## Setup Requirements for Stereo Vision 3D Point Cloud Project

This document outlines all required dependencies and installations for Ubuntu, Windows, and macOS systems.

## System Requirements

### Minimum Hardware

- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**:
  - **NVIDIA**: GTX 1060 or better with CUDA support
  - **AMD**: RX 560 or better with ROCm support
  - **Intel**: Integrated graphics (CPU-only mode)
- **Storage**: 2GB free space for application and dependencies

## Ubuntu 20.04+ / Debian-based Systems

### 1. System Updates

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Essential Build Tools

```bash
sudo apt install -y build-essential cmake git pkg-config
```

### 3. C++ Compiler

```bash
# GCC 9+ (usually pre-installed)
sudo apt install -y gcc g++

# Or Clang (alternative)
sudo apt install -y clang clang++
```

### 4. GPU Support

#### For NVIDIA GPUs (CUDA)

```bash
# Install CUDA Toolkit 11.0+
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

#### For AMD GPUs (ROCm/HIP)

```bash
# Install ROCm 5.0+
wget https://repo.radeon.com/amdgpu-install/5.7.3/ubuntu/jammy/amdgpu-install_5.7.3.50700-1_all.deb
sudo apt install -y ./amdgpu-install_5.7.3.50700-1_all.deb
sudo amdgpu-install --usecase=hiplibsdk,rocm
```

### 5. OpenCV 4.5+

```bash
sudo apt install -y libopencv-dev python3-opencv
```

### 6. PCL (Point Cloud Library) 1.12+

```bash
sudo apt install -y libpcl-dev
```

### 7. VTK (Visualization Toolkit)

```bash
sudo apt install -y libvtk9-dev
```

### 8. Qt6 Development Libraries

```bash
sudo apt install -y qt6-base-dev qt6-base-dev-tools qt6-tools-dev
sudo apt install -y qt6-opengl-dev qt6-opengl-widgets-dev
```

### 9. Additional Dependencies

```bash
sudo apt install -y libboost-all-dev libeigen3-dev
sudo apt install -y libglew-dev libglu1-mesa-dev
sudo apt install -y libmpi-dev openmpi-bin
```

### 10. Optional: Development Tools

```bash
sudo apt install -y valgrind gdb
sudo apt install -y clang-format clang-tidy
```

## Windows 10/11

### 1. Visual Studio 2019/2022

- Download and install [Visual Studio Community](https://visualstudio.microsoft.com/downloads/)
- Include: C++ CMake tools, Windows 10/11 SDK

### 2. CMake

- Download from [cmake.org](https://cmake.org/download/)
- Add to PATH during installation

### 3. Git

- Download from [git-scm.com](https://git-scm.com/download/win)

### 4. GPU Support

#### For NVIDIA GPUs (CUDA)

- Download and install [CUDA Toolkit 11.0+](https://developer.nvidia.com/cuda-downloads)
- Download and install [cuDNN](https://developer.nvidia.com/cudnn) (requires NVIDIA account)

#### For AMD GPUs (ROCm/HIP)

- ROCm support on Windows is limited
- Consider using WSL2 with Ubuntu for AMD GPU development

### 5. Dependencies via vcpkg (Recommended)

```cmd
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install dependencies
.\vcpkg install opencv4[core,highgui,imgproc,calib3d]
.\vcpkg install pcl[core,io,visualization]
.\vcpkg install qt6-base qt6-opengl qt6-opengl-widgets
.\vcpkg install boost eigen3
```

### 6. Alternative: Manual Installation

- **OpenCV**: Download from [opencv.org](https://opencv.org/releases/)
- **PCL**: Download from [pointclouds.org](https://pointclouds.org/downloads/)
- **Qt6**: Download from [qt.io](https://www.qt.io/download)

## macOS 10.15+ (Catalina+)

### 1. Xcode Command Line Tools

```bash
xcode-select --install
```

### 2. Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. GPU Support

#### For Apple Silicon (M1/M2)

- Metal Performance Shaders (MPS) support is built-in
- No additional installation needed

#### For Intel Macs

- OpenCL support is built-in
- No additional installation needed

### 4. Dependencies via Homebrew

```bash
# Core dependencies
brew install cmake git pkg-config

# OpenCV
brew install opencv

# PCL
brew install pcl

# Qt6
brew install qt@6

# Additional libraries
brew install boost eigen glew
```

### 5. Optional: Development Tools

```bash
brew install llvm
brew install --cask visual-studio-code
```

## Platform-Specific Notes

### Ubuntu/Debian

- **VTK Symlink**: The build system automatically creates a symlink from `/usr/include/vtk-9.1` to `/usr/include/vtk` for PCL compatibility
- **GPU Groups**: Add user to `video` and `render` groups for GPU access:

  ```bash
  sudo usermod -a -G video,render $USER
  # Logout and login again for changes to take effect
  ```

### Windows

- **PATH Environment**: Ensure CUDA, CMake, and Git are in your system PATH
- **Visual Studio**: Use "Developer Command Prompt" or "x64 Native Tools Command Prompt"
- **WSL2**: Consider using WSL2 with Ubuntu for better Linux compatibility

### macOS

- **Xcode**: Full Xcode installation may be required for some dependencies
- **Rosetta**: Install Rosetta 2 for Intel Macs: `softwareupdate --install-rosetta`
- **Homebrew Path**: Ensure Homebrew is in your PATH (usually `/opt/homebrew/bin` on Apple Silicon)

## Verification Commands

### Check GPU Support

```bash
# NVIDIA
nvidia-smi

# AMD
rocm-smi

# Check CUDA
nvcc --version

# Check HIP
hipcc --version
```

### Check Dependencies

```bash
# OpenCV
pkg-config --modversion opencv4

# PCL
pkg-config --modversion pcl_common

# Qt6
qmake6 --version

# CMake
cmake --version
```

## Troubleshooting

### Common Issues

1. **VTK Headers Not Found (Linux)**

   ```bash
   sudo ln -sf /usr/include/vtk-9.1 /usr/include/vtk
   ```

2. **GPU Not Detected**
   - Check user groups: `groups`
   - Verify driver installation
   - Restart system after GPU driver installation

3. **CMake Configuration Errors**
   - Ensure all dependencies are installed
   - Check CMake version (3.18+ required)
   - Clear CMake cache: `rm -rf build_*`

4. **Qt6 Not Found**
   - Install Qt6 development packages
   - Set `CMAKE_PREFIX_PATH` to Qt6 installation directory

### Getting Help

- Check the project's `cmake_build.log` for detailed error messages
- Ensure all dependencies are properly installed
- Verify GPU drivers are up to date
- Check system compatibility with required versions

## Next Steps

After installing all dependencies:

1. Clone the repository
2. Run the appropriate build script:
   - Linux: `./build_amd.sh` or `./build.sh`
   - Windows: Use CMake GUI or command line
   - macOS: `./build.sh`

3. Test the build:

   ```bash
   ./build_amd/stereo_vision_app --help
   ```

4. Run sample data processing:

   ```bash
   ./scripts/download_sample_data.sh
   ./build_amd/stereo_vision_app --console --left data/sample_images/left/ --right data/sample_images/right/
   ```
