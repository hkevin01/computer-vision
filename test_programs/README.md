# Test Programs

This directory contains standalone C++ test programs for various components of the stereo vision project. These programs are used for integration testing and validation of specific features.

## Building and Running

To build and run all tests, use the main `run.sh` script from the project root:
```bash
cd /home/kevin/Projects/computer-vision
./run.sh --tests
```

To run a specific test program after building, you can execute it directly from the `build/` directory:
```bash
# Example for test_camera_manager
./build/test_camera_manager
```

## Test Files

- `test_benchmarking.cpp`: Performance benchmarking tests.
- `test_camera_manager.cpp`: Tests for the main camera management system.
- `test_camera_manager_simple.cpp`: Simplified tests for basic camera functionality.
- `test_cameras.cpp`: General camera hardware tests.
- `test_direct_camera.cpp`: Tests for direct, low-level camera access.
- `test_gui_camera_detection.cpp`: Tests for camera detection within the GUI.
- `test_multicamera.cpp`: Tests for the multi-camera synchronization and capture system.
- `test_neural_network.cpp`: Tests for the neural network stereo matching implementation.
- `test_qt.cpp`: Basic tests for Qt integration.
- `test_single_camera.cpp`: Tests for single-camera operation.
- `priority2_summary.cpp`: Summary and validation for completed Priority 2 features.
- `test_priority2_complete.cpp`: Additional tests for Priority 2 feature integration.
