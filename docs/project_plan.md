# Stereo Vision 3D Point Cloud Project Plan

## Project Overview

**Objective**: Develop a high-performance C++ application that processes stereo camera images to generate accurate 3D point clouds using CUDA acceleration.

**Target Applications**:
- 3D reconstruction from stereo images
- Depth estimation for robotics
- Augmented reality applications
- Industrial 3D scanning

## Technical Architecture

### Core Technologies
- **Language**: C++17
- **GPU Computing**: CUDA 11.0+ (NVIDIA) / ROCm 5.0+ with HIP (AMD)
- **Computer Vision**: OpenCV 4.5+
- **Point Cloud Processing**: PCL (Point Cloud Library) 1.12+
- **GUI Framework**: Qt6 (with Qt5 fallback)
- **Build System**: CMake 3.18+

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Input  │───▶│  Preprocessing  │───▶│   Calibration   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Point Cloud    │◀───│ Stereo Matching │◀───│  Rectification  │
│   Generation    │    │ (CUDA/HIP GPU)  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│ Visualization & │    │   Export/Save   │
│      GUI        │    │   (PLY, PCD)    │
└─────────────────┘    └─────────────────┘
```

## Development Phases

### Phase 1: Foundation Setup (Week 1-2)
**Objectives**: Establish project infrastructure and basic components

#### Tasks:
- [x] Project structure creation
- [x] CMake build system setup
- [x] Dependencies configuration
- [x] Test plan documentation
- [ ] Basic camera interface implementation
- [ ] Image loading and basic preprocessing
- [ ] Unit test framework setup

**Deliverables**:
- Compiling project with all dependencies
- Basic image I/O functionality
- Camera interface classes

### Phase 2: Camera Calibration (Week 3-4)
**Objectives**: Implement robust stereo camera calibration

#### Tasks:
- [ ] Checkerboard detection algorithm
- [ ] Single camera calibration
- [ ] Stereo camera calibration
- [ ] Calibration data persistence
- [ ] Calibration accuracy validation

**Deliverables**:
- Camera calibration module
- Calibration data format specification
- Calibration accuracy metrics

### Phase 3: Stereo Vision Core (Week 5-7)
**Objectives**: Develop stereo matching and depth estimation

#### Tasks:
- [ ] Image rectification implementation
- [ ] Semi-Global Block Matching (SGBM) algorithm
- [ ] CUDA kernel for stereo matching acceleration
- [ ] HIP kernel for AMD GPU acceleration (alternative to CUDA)
- [ ] Disparity map generation
- [ ] Disparity post-processing (median filter, speckle removal)

**Deliverables**:
- CPU stereo matching implementation
- CUDA-accelerated stereo matching (NVIDIA)
- HIP-accelerated stereo matching (AMD)
- Disparity map quality assessment

### Phase 4: Point Cloud Generation (Week 8-9)
**Objectives**: Convert disparity maps to 3D point clouds

#### Tasks:
- [ ] 3D point reconstruction from disparity
- [ ] Point cloud filtering and noise reduction
- [ ] Color mapping from original images
- [ ] PCL integration for point cloud operations
- [ ] Multiple export format support (PLY, PCD, XYZ)

**Deliverables**:
- 3D point cloud generation pipeline
- Point cloud quality metrics
- Export functionality

### Phase 5: GUI Development (Week 10-11)
**Objectives**: Create user-friendly interface

#### Tasks:
- [ ] Qt6 main window design with modern UI
- [ ] Image display widgets with zoom/pan functionality
- [ ] Parameter adjustment controls with real-time updates
- [ ] Real-time preview functionality
- [ ] 3D point cloud visualization widget (Qt3D or VTK integration)
- [ ] File management interface with drag-and-drop support
- [ ] Settings dialog for calibration and algorithm parameters

**Deliverables**:
- Complete GUI application
- User manual and documentation
- GUI responsiveness optimization

### Phase 6: Optimization & Testing (Week 12-13)
**Objectives**: Performance optimization and comprehensive testing

#### Tasks:
- [ ] CUDA kernel optimization
- [ ] HIP kernel optimization for AMD GPUs
- [ ] Memory usage optimization
- [ ] Multi-threading implementation
- [ ] Comprehensive unit testing
- [ ] Integration testing
- [ ] Performance benchmarking

**Deliverables**:
- Performance benchmarks
- Test coverage report
- Optimization documentation

### Phase 7: Documentation & Deployment (Week 14)
**Objectives**: Finalize documentation and prepare for deployment

#### Tasks:
- [x] Test plan documentation
- [ ] API documentation completion
- [ ] User guide creation
- [ ] Installation instructions
- [ ] Sample data preparation
- [ ] CI/CD pipeline setup
- [ ] Release preparation

**Deliverables**:
- Complete documentation
- Installation packages
- Sample datasets

## Immediate Next Steps

Following the successful project setup and AMD GPU build, the immediate next steps are:

- [x] Verify AMD ROCm/HIP build stability and basic GPU kernels performance
- [x] Implement Phase 1 core components:
  - Basic camera interface and stereo image I/O
  - Initial preprocessing routines (undistort, rectify) 
  - Unit tests framework with GoogleTest
- [x] Create comprehensive test suite with both core and GUI tests
- [x] Enhance run.sh script with build options and test execution
- [ ] Create a sample application function to load stereo images, compute disparity, and visualize results
- [ ] Extend CI/CD pipeline to include AMD and NVIDIA GPU build & test jobs
- [ ] Update quick start guide and SETUP_COMPLETE.md with current instructions
- [x] Refine CMake configurations to compile HIP GPU kernels correctly (use hipcc or enable blockDim/threadIdx macros)

## Technical Specifications

### Performance Requirements
- **Real-time Processing**: Target 30 FPS for 640x480 stereo pairs
- **Accuracy**: Sub-millimeter precision at 1-meter distance
- **GPU Memory**: Efficient CUDA memory management
- **Scalability**: Support for various image resolutions

### Algorithm Specifications

#### Stereo Matching Algorithm
- **Primary Method**: Semi-Global Block Matching (SGBM)
- **Block Size**: Configurable 5x5 to 21x21
- **Disparity Range**: 0-256 pixels
- **Post-processing**: Median filtering, speckle removal, hole filling

#### CUDA/HIP Optimization
- **Kernel Design**: Optimized for Tesla/RTX (NVIDIA) and RDNA/CDNA (AMD) architectures
- **Memory Pattern**: Coalesced global memory access
- **Shared Memory**: Efficient use for block matching
- **Streams**: Asynchronous processing pipeline
- **Cross-Platform**: Unified codebase using HIP for portability

### Data Formats

#### Calibration Data
```cpp
struct CameraParameters {
    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
    cv::Size image_size;
};

struct StereoParameters {
    CameraParameters left_camera;
    CameraParameters right_camera;
    cv::Mat R, T, E, F;
    cv::Mat R1, R2, P1, P2, Q;
};
```

#### Point Cloud Format
- **Internal**: PCL PointXYZRGB
- **Export**: PLY (binary/ASCII), PCD, XYZ

## Resource Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with Compute Capability 7.5+ OR AMD GPU with ROCm support
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 500MB for application, additional for datasets

### Software Dependencies
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+
- **Compiler**: GCC 9+ / MSVC 2019+ / Clang 10+
- **GPU Runtime**: CUDA Toolkit 11.0+ (NVIDIA) / ROCm 5.0+ (AMD)
- **OpenCV**: 4.5+
- **PCL**: 1.12+

## Risk Assessment

### Technical Risks
1. **GPU Compatibility**: Different GPU architectures (NVIDIA vs AMD)
   - *Mitigation*: HIP abstraction layer for cross-platform GPU support
2. **Performance Bottlenecks**: Real-time processing requirements
   - *Mitigation*: Profiling and optimization phases
3. **Calibration Accuracy**: Poor calibration affecting results
   - *Mitigation*: Robust calibration validation

### Project Risks
1. **Dependency Issues**: Complex library dependencies
   - *Mitigation*: Docker containerization option
2. **Platform Compatibility**: Multi-platform support
   - *Mitigation*: CI/CD testing on multiple platforms

## Success Metrics

### Functional Metrics
- [ ] Successful stereo camera calibration with <0.5 pixel reprojection error
- [ ] Real-time stereo matching at 30 FPS for 640x480 images
- [ ] Point cloud generation with <1% outliers
- [ ] GUI responsiveness <100ms for parameter changes

### Quality Metrics
- [ ] Code coverage >90%
- [ ] Memory leaks: 0 detected
- [ ] CUDA kernel efficiency >80% theoretical peak (NVIDIA)
- [ ] HIP kernel efficiency >80% theoretical peak (AMD)
- [ ] User acceptance testing score >4/5

## Future Enhancements

### Version 2.0 Features
- Deep learning-based stereo matching
- Multi-view stereo reconstruction
- Real-time SLAM integration
- Cloud processing capabilities
- Mobile device support

### Research Integration
- Integration with latest stereo vision research
- Machine learning model deployment
- Advanced filtering techniques
- Real-time optimization algorithms

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Foundation | 2 weeks | Working build system |
| Calibration | 2 weeks | Camera calibration module |
| Stereo Vision | 3 weeks | CUDA stereo matching |
| Point Cloud | 2 weeks | 3D reconstruction pipeline |
| GUI | 2 weeks | Complete user interface |
| Optimization | 2 weeks | Performance benchmarks |
| Documentation | 1 week | Release package |

**Total Duration**: 14 weeks
**Target Release**: Q2 2025
