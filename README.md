# Stereo Vision 3D Point Cloud Generator

A high-performance C++ application for generating 3D point clouds from stereo camera images using GPU acceleration (CUDA for NVIDIA or HIP for AMD GPUs).

## Features

- **Stereo Camera Calibration**: Automatic camera calibration using checkerboard patterns
- **Real-time Stereo Vision**: GPU-accelerated stereo matching algorithms
- **Cross-Platform GPU Support**: NVIDIA CUDA and AMD HIP backends
- **3D Point Cloud Generation**: Convert disparity maps to dense point clouds
- **Interactive GUI**: User-friendly interface for parameter tuning and visualization
- **Multiple Export Formats**: Support for PLY, PCD, and other point cloud formats

## GPU Support

This project supports both NVIDIA and AMD GPUs:

- **NVIDIA GPUs**: Uses CUDA for acceleration
- **AMD GPUs**: Uses ROCm/HIP for acceleration  
- **CPU Fallback**: Automatic fallback to CPU-only mode if no GPU is detected

## Dependencies

### Required Libraries
- **OpenCV** (>= 4.5): Computer vision and image processing
- **PCL** (Point Cloud Library >= 1.12): Point cloud processing and visualization
- **Qt6** (>= 6.0): GUI framework
- **VTK** (>= 9.0): Visualization toolkit (dependency of PCL)
- **CMake** (>= 3.18): Build system

### GPU Runtime (Optional)
- **NVIDIA**: CUDA Toolkit (>= 11.0)
- **AMD**: ROCm (>= 5.0) with HIP support

## Installation

**📋 Complete setup instructions for Ubuntu, Windows, and macOS are available in [docs/SETUP_REQUIREMENTS.md](docs/SETUP_REQUIREMENTS.md)**

### Quick Setup (Ubuntu/Debian)

#### For NVIDIA GPUs:
```bash
# Run the main setup script (auto-detects NVIDIA)
./setup_dev_environment.sh
```

#### For AMD GPUs:
```bash
# First run basic setup
./setup_dev_environment.sh

# Then run AMD-specific setup
./setup_amd_gpu.sh
```

#### Manual Installation (Ubuntu):
```bash
# Install OpenCV
sudo apt update
sudo apt install libopencv-dev

# Install PCL and VTK
sudo apt install libpcl-dev libvtk9-dev

# Install Qt6
sudo apt install qt6-base-dev qt6-opengl-dev qt6-opengl-widgets-dev

# Install additional dependencies
sudo apt install libboost-all-dev libeigen3-dev libglew-dev

# For NVIDIA: Install CUDA (follow NVIDIA's official guide)
# For AMD: Install ROCm (see setup_amd_gpu.sh)
```

## Building

### Quick Build (Auto-detection)
```bash
# Auto-detects GPU and builds accordingly
./build.sh
```

### Manual Build with GPU Backend Selection

#### For NVIDIA GPUs:
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_HIP=OFF
make -j$(nproc)
```

#### For AMD GPUs:
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF -DUSE_HIP=ON
make -j$(nproc)
```

#### CPU-only:
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF -DUSE_HIP=OFF
make -j$(nproc)
```

## Usage

### Running the Application
```bash
./stereo_vision_app
```

### Camera Calibration
1. Print a checkerboard pattern (9x6 recommended)
2. Capture calibration images with both cameras
3. Use the calibration tool to compute camera parameters

### Point Cloud Generation
1. Load calibrated camera parameters
2. Capture or load stereo image pairs
3. Adjust stereo matching parameters
4. Generate and export point cloud

## Project Structure

```
├── src/
│   ├── core/           # Core stereo vision algorithms
│   ├── cuda/           # CUDA kernels and GPU processing
│   ├── gui/            # GTK3 user interface
│   ├── utils/          # Utility functions
│   └── main.cpp        # Application entry point
├── include/            # Header files
├── data/              # Sample data and test images
├── tests/             # Unit tests
├── docs/              # Documentation
└── scripts/           # Build and utility scripts
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
