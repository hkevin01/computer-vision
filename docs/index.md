# Computer Vision Stereo Processing Library

Welcome to the comprehensive documentation for the Computer Vision Stereo Processing Library - a high-performance C++ library for stereo image processing, 3D reconstruction, and real-time computer vision applications.

## 🎯 What is This Library?

This library provides a complete toolkit for:

- **Stereo Camera Calibration**: Advanced calibration algorithms with subpixel accuracy
- **Real-time Stereo Matching**: GPU-accelerated disparity map generation
- **3D Point Cloud Generation**: High-quality depth reconstruction
- **Multi-Camera Systems**: Synchronized multi-camera processing
- **Streaming Optimization**: Advanced buffering and performance optimization
- **AI-Enhanced Processing**: Neural stereo matching and depth estimation

## ✨ Key Features

- **High Performance**: Optimized C++ code with CUDA/HIP GPU acceleration
- **Real-time Processing**: Streaming pipelines with adaptive frame rate control
- **Modern Architecture**: Thread-safe, modular design with structured logging
- **Cross-platform**: Linux, Windows, and macOS support
- **Comprehensive APIs**: Easy-to-use interfaces for both beginners and experts
- **Rich GUI Tools**: Qt-based calibration wizards and live tuning interfaces

## 🚀 Quick Start

=== "Ubuntu/Debian"

    ```bash
    # Install dependencies
    sudo apt update
    sudo apt install libopencv-dev libpcl-dev qt5-default cmake build-essential

    # Clone and build
    git clone https://github.com/computer-vision-project/computer-vision.git
    cd computer-vision
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    ```

=== "Docker"

    ```bash
    # Pull and run the container
    docker pull computer-vision/stereo-processing:latest
    docker run -it --rm -v /dev/video0:/dev/video0 computer-vision/stereo-processing
    ```

=== "From Source"

    ```bash
    # Detailed build instructions
    git clone https://github.com/computer-vision-project/computer-vision.git
    cd computer-vision
    ./scripts/setup_dev_environment.sh
    ./build.sh
    ```

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 20.04, Windows 10 | Ubuntu 22.04, Windows 11 |
| **CPU** | Intel i5-8000 series, AMD Ryzen 5 | Intel i7-10000 series, AMD Ryzen 7 |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | NVIDIA GTX 1060, AMD RX 580 | NVIDIA RTX 3070, AMD RX 6700 XT |
| **OpenCV** | 4.5.0 | 4.8.0+ |
| **PCL** | 1.12.0 | 1.14.0+ |

## 🏗 Architecture Overview

    graph TB
        A[Camera Input] --> B[Calibration Module]
        A --> C[Stereo Matcher]
        B --> C
        C --> D[Point Cloud Processor]
        C --> E[Streaming Optimizer]
        D --> F[3D Visualization]
        E --> G[Real-time Display]

        H[AI Module] --> C
        I[Multi-Camera System] --> B
        I --> C

        style A fill:#e1f5fe
        style B fill:#f3e5f5
        style C fill:#e8f5e8
        style D fill:#fff3e0
        style E fill:#fce4ec

## 🎯 Use Cases

### Research & Development

- Computer vision research projects
- Robotics and autonomous systems
- Augmented reality applications
- 3D reconstruction pipelines

### Industrial Applications

- Quality control and inspection
- 3D measurement and metrology
- Robotic guidance systems
- Automated manufacturing

### Consumer Applications

- Gaming and entertainment
- Mobile 3D scanning
- Virtual reality systems
- Gesture recognition

## 📚 Documentation Structure

- **[Getting Started](getting-started/quick-start.md)**: Installation and setup guides
- **[User Guide](user-guide/overview.md)**: Comprehensive usage documentation
- **[API Reference](api/core.md)**: Detailed API documentation
- **[Tutorials](tutorials/basic-stereo.md)**: Step-by-step tutorials
- **[Development](development/contributing.md)**: Contributing and development guides

## 🤝 Community & Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/computer-vision-project/computer-vision/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/computer-vision-project/computer-vision/discussions)
- **Stack Overflow**: Ask questions with the `computer-vision-stereo` tag

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/computer-vision-project/computer-vision/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

Built with modern C++ and powered by:

- [OpenCV](https://opencv.org/) - Computer Vision Library
- [PCL](https://pointclouds.org/) - Point Cloud Library
- [Qt](https://www.qt.io/) - Cross-platform GUI Framework
- [spdlog](https://github.com/gabime/spdlog) - Fast C++ logging library

---

!!! tip "New to stereo vision?"
    Start with our [Basic Stereo Setup Tutorial](tutorials/basic-stereo.md) to learn the fundamentals of stereo camera systems and 3D reconstruction.

!!! info "Performance tip"
    For the best real-time performance, check out our [Streaming Optimization Guide](user-guide/streaming.md) to learn about advanced buffering and GPU acceleration.
