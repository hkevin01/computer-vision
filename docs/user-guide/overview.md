# User Guide Overview

Welcome to the comprehensive user guide for the Computer Vision Stereo Processing Library. This guide covers everything you need to know to effectively use the library for your stereo vision projects.

## What You'll Learn

This user guide is organized into logical sections that build upon each other:

### Core Concepts
- **[Stereo Vision Fundamentals](stereo-fundamentals.md)**: Understanding stereo vision principles
- **[Camera Calibration](calibration.md)**: Calibrating your stereo camera setup
- **[Disparity Mapping](disparity.md)**: Generating depth maps from stereo pairs
- **[Point Clouds](point-clouds.md)**: Converting disparity to 3D point clouds

### Processing Pipelines
- **[Real-time Processing](streaming.md)**: Streaming optimization and performance
- **[Batch Processing](batch.md)**: Processing large datasets efficiently
- **[Multi-camera Systems](multi-camera.md)**: Working with multiple camera pairs

### Advanced Features
- **[AI Enhancement](ai-features.md)**: Neural stereo matching and depth estimation
- **[GPU Acceleration](gpu.md)**: CUDA and HIP optimization
- **[Custom Algorithms](custom-algorithms.md)**: Implementing custom processing

### Tools and Applications
- **[GUI Applications](gui-tools.md)**: Using the Qt-based tools
- **[Command Line Tools](cli-tools.md)**: Batch processing and automation
- **[Integration Guide](integration.md)**: Using the library in your projects

## Library Architecture

The Computer Vision Stereo Processing Library is built with a modular architecture:

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                   │
├─────────────────────────────────────────────────────┤
│  GUI Tools  │  CLI Apps  │  Examples  │  Your App  │
├─────────────────────────────────────────────────────┤
│                   Core Library                      │
├─────────────────────────────────────────────────────┤
│ Calibration │ Stereo    │ Point     │ Streaming    │
│ Module      │ Matching  │ Cloud     │ Optimizer    │
├─────────────────────────────────────────────────────┤
│              Hardware Abstraction                   │
├─────────────────────────────────────────────────────┤
│   OpenCV    │   PCL     │   CUDA    │     Qt5      │
└─────────────────────────────────────────────────────┘
```

### Key Components

#### Core Library (`stereo_vision_core`)
The main processing library containing:

- **`StereoProcessor`**: Main stereo processing class
- **`CalibrationManager`**: Camera calibration and parameter management
- **`StreamingOptimizer`**: Real-time performance optimization
- **`PointCloudProcessor`**: 3D reconstruction and processing

#### GUI Library (`stereo_vision_gui`)
Qt-based graphical interface components:

- **Live tuning interfaces**: Real-time parameter adjustment
- **Calibration wizards**: Step-by-step calibration guides
- **Visualization tools**: 3D point cloud viewers

#### Support Libraries
- **Logging**: Structured logging with configurable levels
- **Configuration**: YAML-based configuration management
- **Utilities**: Common helper functions and data structures

## Basic Workflow

A typical stereo vision workflow follows these steps:

### 1. Camera Setup and Calibration

    # Connect stereo cameras
    # Run calibration wizard
    ./build/live_stereo_tuning

### 2. Configure Processing Parameters

    # Edit configuration file
    nano config/stereo_config.yaml

### 3. Process Images

    # Single image pair
    ./build/stereo_vision_app_simple \
        --left image_left.jpg \
        --right image_right.jpg \
        --output disparity.png

### 4. Generate Point Clouds

    # Convert disparity to 3D points
    ./build/point_cloud_processor \
        --disparity disparity.png \
        --calibration calibration.yaml \
        --output points.ply

## Configuration System

The library uses a hierarchical configuration system:

### Global Configuration
Located in `config/stereo_config.yaml`:

    stereo:
      algorithm: "sgbm"
      block_size: 5
      min_disparity: 0
      max_disparity: 96

    calibration:
      board_size: [9, 6]
      square_size: 25.0

    streaming:
      buffer_size: 10
      max_fps: 30
      adaptive_quality: true

### Runtime Configuration
Parameters can be overridden at runtime:

    StereoProcessor processor;
    processor.setParameter("stereo.block_size", 7);
    processor.setParameter("stereo.max_disparity", 128);

### Environment Variables
Some settings can be controlled via environment variables:

    export STEREO_LOG_LEVEL=debug
    export STEREO_GPU_DEVICE=0
    export STEREO_THREADS=8

## Performance Considerations

### Hardware Requirements

For optimal performance, consider:

- **CPU**: Multi-core processor (Intel i7, AMD Ryzen 7+)
- **RAM**: 16GB+ for high-resolution processing
- **GPU**: NVIDIA RTX series or AMD RX 6000+ for acceleration
- **Storage**: SSD for fast image I/O

### Optimization Settings

#### Real-time Processing

    streaming:
      enable_gpu: true
      buffer_size: 5
      adaptive_fps: true
      frame_drop_threshold: 0.8

#### High Quality Processing

    stereo:
      algorithm: "sgbm"
      block_size: 3
      speckle_filtering: true
      post_processing: true

#### Batch Processing

    batch:
      parallel_jobs: 8
      chunk_size: 100
      memory_limit: "8GB"

## Error Handling and Debugging

### Logging Configuration

The library provides comprehensive logging:

    # Set log level (trace, debug, info, warn, error, critical)
    export STEREO_LOG_LEVEL=debug

    # Enable file logging
    export STEREO_LOG_FILE=/tmp/stereo_debug.log

### Common Issues

#### Camera Not Detected

    # Check camera permissions
    ls -la /dev/video*

    # Test camera access
    ./build/test_camera_detection

#### Poor Calibration Results

    # Use more calibration images (20+ recommended)
    # Ensure good lighting and sharp images
    # Vary pose angles and distances

#### Slow Performance

    # Enable GPU acceleration
    # Reduce image resolution
    # Adjust processing parameters
    # Use streaming optimization

### Debug Tools

The library includes several debugging utilities:

    # System diagnostics
    ./scripts/diagnose_env.sh

    # Camera testing
    ./build/test_camera_detection

    # Performance benchmarking
    ./build/benchmark_app

## Getting Help

### Documentation Resources

- **[API Reference](../api/core.md)**: Detailed API documentation
- **[Tutorials](../tutorials/basic-stereo.md)**: Step-by-step guides
- **[Examples](examples.md)**: Code examples and use cases
- **[FAQ](faq.md)**: Frequently asked questions

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A and sharing
- **Stack Overflow**: Technical questions with `computer-vision-stereo` tag

### Professional Support

For commercial applications or custom development:

- **Consulting Services**: Expert guidance and implementation
- **Training Workshops**: Team training and best practices
- **Custom Development**: Tailored solutions for specific needs

---

!!! tip "Getting Started"
    New to stereo vision? Start with [Stereo Vision Fundamentals](stereo-fundamentals.md) to understand the core concepts.

!!! info "Performance"
    Looking for optimal performance? Check out our [Streaming Optimization Guide](streaming.md) for real-time processing tips.

!!! warning "Hardware"
    Make sure your hardware meets the [minimum requirements](../getting-started/quick-start.md#system-requirements) for smooth operation.
