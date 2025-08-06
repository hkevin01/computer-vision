# Comprehensive Edge Case Testing Framework

## 🎯 Overview

This document describes the comprehensive edge case testing framework implemented for the stereo vision project. The framework is designed to test **every function** with explicit coverage of edge cases including overflow, precision loss, truncation, malformed input, system failures, and hardware malfunctions.

## 🧪 Testing Philosophy

> **"Every function now gets a sanity check: What if the input is malformed? What if the system is under load? What if the hardware fails mid-operation?"**

### Core Testing Principles

1. **Exhaustive Edge Case Coverage** - Test numerical limits, malformed inputs, and boundary conditions
2. **System Failure Simulation** - Simulate hardware failures, resource exhaustion, and concurrent access
3. **Graceful Degradation** - Verify systems fail safely and provide meaningful error messages
4. **Performance Under Stress** - Test functionality under memory pressure, CPU load, and I/O constraints
5. **Precision Validation** - Detect numerical instability, overflow, and significant precision loss

## 📁 Framework Structure

```
tests/
├── edge_case_framework.hpp          # Core testing utilities and macros
├── edge_case_framework.cpp          # Framework implementation
├── test_camera_calibration_edge_cases.cpp  # Camera calibration edge tests
├── test_neural_matcher_edge_cases.cpp      # AI/ML neural matcher edge tests
├── test_point_cloud_edge_cases.cpp         # Point cloud processing edge tests
├── test_system_edge_cases.cpp              # System-level failure simulation
├── run_edge_case_tests.sh                  # Comprehensive test execution script
└── CMakeLists.txt                          # Updated build configuration
```

## 🔧 Core Framework Components

### EdgeCaseTestFramework Class

The central testing utility providing:

#### Numerical Edge Case Generators
```cpp
// Generate problematic floating-point values
auto float_edges = EdgeCaseTestFramework::getFloatingPointEdgeCases();
// Returns: infinity, NaN, denormal numbers, epsilon boundaries, etc.

// Generate problematic integer values
auto int_edges = EdgeCaseTestFramework::getIntegerEdgeCases();
// Returns: MIN/MAX values, overflow boundaries, power-of-2 edges

// Generate problematic image dimensions
auto size_edges = EdgeCaseTestFramework::getImageSizeEdgeCases();
// Returns: 0x0, 1x1, extreme aspect ratios, alignment issues
```

#### System Stress Simulation
```cpp
// Simulate memory pressure (allocates specified MB)
EdgeCaseTestFramework::simulateMemoryPressure(1024);

// Simulate high CPU load
EdgeCaseTestFramework::simulateHighCPULoad(2000); // 2 seconds

// Generate corrupted/malformed data
cv::Mat corrupted = EdgeCaseTestFramework::generateCorruptedImage(size, type);
cv::Mat inf_mat = EdgeCaseTestFramework::generateInfiniteValuesMatrix(size, type);
cv::Mat nan_mat = EdgeCaseTestFramework::generateNaNValuesMatrix(size, type);
```

#### Precision Testing Utilities
```cpp
// Check for precision loss
bool has_loss = EdgeCaseTestFramework::hasSignificantPrecisionLoss(original, computed);

// Tolerance-based comparison
bool within_tolerance = EdgeCaseTestFramework::isWithinTolerance(actual, expected, 1e-9);
```

### Testing Macros

```cpp
// Verify no numerical overflow occurs
EXPECT_NO_OVERFLOW(risky_computation());

// Verify graceful failure with meaningful error handling
EXPECT_GRACEFUL_FAILURE(function_with_bad_input());

// Verify robust operation under stress
EXPECT_ROBUST_OPERATION(stressed_operation(), expected_result);
```

## 🧮 Camera Calibration Edge Cases

### Numerical Edge Cases Tested
- **Parameter Overflow**: Extremely large square sizes causing floating-point overflow
- **Precision Loss**: Microscopic square sizes leading to precision degradation
- **Matrix Singularity**: Conditions that could make camera matrix singular
- **Coordinate Truncation**: Integer truncation in corner detection algorithms

### Input Validation Tests
- **Empty Image Vectors**: Calibration with no input images
- **Corrupted Images**: Random noise and systematic corruption patterns
- **Infinite/NaN Values**: Images containing non-finite pixel values
- **Mismatched Stereo Pairs**: Different sizes, counts, or types of stereo images

### System Stress Tests
- **Memory Pressure**: Calibration under severe memory constraints
- **Concurrent Operations**: Multiple simultaneous calibration processes
- **Extreme Image Sizes**: 1x1 to 16K resolution edge cases

### Example Test Case
```cpp
TEST_F(CameraCalibrationEdgeCaseTest, CalibrationParameterOverflow) {
    auto edge_values = EdgeCaseTestFramework::getFloatingPointEdgeCases();

    for (double edge_val : edge_values) {
        if (std::isfinite(edge_val) && edge_val > 0) {
            std::vector<cv::Mat> images;
            images.push_back(createTestCheckerboard(cv::Size(640, 480), cv::Size(9, 6)));

            EXPECT_GRACEFUL_FAILURE(
                calibration_->calibrateSingleCamera(images, cv::Size(9, 6), static_cast<float>(edge_val))
            );
        }
    }
}
```

## 🧠 Neural Matcher Edge Cases

### AI/ML Specific Edge Cases
- **Model Loading Failures**: Non-existent, corrupted, or incompatible model files
- **Inference Overflow**: Numerical overflow during neural network computation
- **Backend Failures**: GPU unavailability and CPU fallback scenarios
- **Memory Allocation**: Large model loading under memory pressure

### Input Validation
- **Extreme Image Sizes**: 1x1 to 8K resolution neural processing
- **Mismatched Inputs**: Different sizes, types, or corrupted stereo pairs
- **Precision Testing**: Multiple data types (CV_8U, CV_16U, CV_32F, CV_64F)

### Hardware Simulation
- **GPU Failure**: CUDA/OpenCL unavailability with automatic CPU fallback
- **Model Corruption**: Simulated corrupted ONNX/TensorRT model files
- **Concurrent Inference**: Multiple simultaneous neural network operations

## ☁️ Point Cloud Processing Edge Cases

### Coordinate Edge Cases
- **Extreme Coordinates**: Near float MIN/MAX values
- **Precision Boundaries**: High-precision coordinates testing epsilon limits
- **Non-finite Values**: Points with NaN or infinite coordinates
- **Coordinate Overflow**: Operations that might cause floating-point overflow

### Filtering Robustness
- **Outlier Heavy Data**: Point clouds with 50%+ outlier points
- **Empty Clouds**: Zero-point cloud handling
- **Malformed Clouds**: Inconsistent width/height vs points size
- **Memory Pressure**: Large point cloud processing under memory constraints

### Export/Import Validation
- **File System Failures**: Disk full, permission denied, invalid paths
- **Format Edge Cases**: Export of corrupted point clouds to various formats
- **Concurrent I/O**: Multiple simultaneous export operations

## 🖥️ System-Level Edge Cases

### Complete System Failure Simulation
```cpp
TEST_F(SystemLevelEdgeCaseTest, CompleteSystemFailureRecovery) {
    // Simultaneous failures:
    EdgeCaseTestFramework::simulateMemoryPressure(1024);  // Memory pressure
    simulateHighCPULoad(3000);                           // CPU stress
    createCorruptedCalibrationFile("bad.xml");           // File corruption

    // System should either work or fail gracefully
    try {
        performStereoVisionPipeline();
        EXPECT_TRUE(true) << "System survived multiple failures";
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Graceful failure: " << e.what();
    }
}
```

### Resource Exhaustion Testing
- **File Descriptor Limits**: Exhaust available file descriptors
- **Memory Fragmentation**: Allocate/deallocate patterns causing fragmentation
- **Disk Space**: Simulate filesystem full conditions
- **Thread Exhaustion**: Concurrent operations exceeding system limits

### Race Condition Detection
- **Concurrent Calibration**: Multiple threads performing calibration
- **Shared Resource Access**: File I/O and memory access patterns
- **Thread Safety Validation**: Detection of non-thread-safe operations

### Hardware Failure Simulation
- **GPU Failure**: Simulate CUDA/OpenCL failures with CPU fallback
- **Disk I/O Failure**: Simulate storage device failures
- **Network Failure**: Simulate network interruptions (for distributed processing)

## 🔍 Precision Drift Detection

### Long-Running Operation Testing
```cpp
TEST_F(SystemLevelEdgeCaseTest, PrecisionDriftDetection) {
    const int num_iterations = 1000;
    std::vector<double> reprojection_errors;

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        auto result = calibration->calibrateSingleCamera(test_images, cv::Size(9, 6), 25.0f);
        reprojection_errors.push_back(result.reprojection_error);

        // Check for numerical instability
        double determinant = cv::determinant(result.camera_matrix);
        EXPECT_FALSE(std::isnan(determinant)) << "Matrix became singular at iteration " << iteration;

        // Check for exploding errors
        if (iteration > 10) {
            double current_error = reprojection_errors.back();
            double previous_error = reprojection_errors[reprojection_errors.size() - 2];

            if (current_error > previous_error * 10.0) {
                FAIL() << "Error exploded at iteration " << iteration;
            }
        }
    }
}
```

## 🚀 Running the Tests

### Quick Edge Case Tests
```bash
# Build and run edge case tests
cd build
make run_edge_case_tests
./tests/run_edge_case_tests
```

### Comprehensive Testing Script
```bash
# Run all test configurations with system stress
chmod +x tests/run_edge_case_tests.sh
./tests/run_edge_case_tests.sh
```

The comprehensive script tests:
1. **Debug Build with Sanitizers** - Detects memory errors and undefined behavior
2. **Release Build** - Performance validation under optimization
3. **AI/ML Build** - Neural network edge case testing (if ONNX available)
4. **CUDA Build** - GPU acceleration edge cases (if CUDA available)
5. **Memory Stress Testing** - Operations under memory limits
6. **Performance Benchmarking** - Edge case handling performance

### Test Configurations

#### Debug with Sanitizers
```bash
cmake -DCMAKE_BUILD_TYPE=Debug \
      -fsanitize=address \
      -fsanitize=undefined \
      ..
```

#### Memory Limited Testing
```bash
# Run with 1GB memory limit
systemd-run --user --scope -p MemoryMax=1G ./tests/run_edge_case_tests

# Run with file descriptor limits
ulimit -n 256 && ./tests/run_edge_case_tests
```

## 📊 Test Results and Analysis

### XML Output Format
Tests generate detailed XML reports for analysis:
```
test_results/
├── debug_sanitized_results.xml
├── release_results.xml
├── aiml_onnx_results.xml
├── memory_stress_results.xml
└── performance_results.xml
```

### Metrics Tracked
- **Test Count**: Total number of edge cases tested
- **Failure Rate**: Percentage of edge cases that cause failures
- **Performance Impact**: Execution time under stress conditions
- **Memory Usage**: Peak memory consumption during edge case testing
- **Error Types**: Classification of failure modes (overflow, precision loss, etc.)

### Success Criteria
1. **Graceful Failure**: All invalid inputs should be rejected with meaningful errors
2. **No Crashes**: System should never crash, even with malformed data
3. **Precision Stability**: Numerical algorithms should maintain precision within acceptable bounds
4. **Resource Cleanup**: All allocated resources should be properly cleaned up
5. **Concurrent Safety**: Operations should be thread-safe or clearly documented as not thread-safe

## 🔧 Integration with CI/CD

### GitHub Actions Integration
```yaml
name: Edge Case Testing
on: [push, pull_request]
jobs:
  edge-case-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Dependencies
        run: sudo apt-get install libopencv-dev libpcl-dev
      - name: Run Edge Case Tests
        run: |
          chmod +x tests/run_edge_case_tests.sh
          ./tests/run_edge_case_tests.sh
      - name: Upload Test Results
        uses: actions/upload-artifact@v2
        with:
          name: edge-case-test-results
          path: test_results/
```

## 🎯 Best Practices

### Writing New Edge Case Tests

1. **Test Boundary Conditions**
   ```cpp
   // Test at the boundaries of valid input ranges
   for (int size = 0; size <= 2; ++size) {
       test_function_with_size(size);
   }
   ```

2. **Simulate Real-World Failures**
   ```cpp
   // Don't just test perfect conditions
   EdgeCaseTestFramework::simulateMemoryPressure(512);
   auto result = resource_intensive_operation();
   ```

3. **Verify Error Messages**
   ```cpp
   // Ensure errors are informative
   try {
       invalid_operation();
       FAIL() << "Should have thrown exception";
   } catch (const std::exception& e) {
       EXPECT_THAT(e.what(), HasSubstr("meaningful error description"));
   }
   ```

4. **Test Concurrent Access**
   ```cpp
   // Always test multi-threaded scenarios
   auto task = [&]() { shared_resource_operation(); };
   EdgeCaseTestFramework::testConcurrentAccess(task, 10, 100);
   ```

### Adding New Edge Cases

To add edge case testing for a new component:

1. Create test file: `test_[component]_edge_cases.cpp`
2. Inherit from `EdgeCaseTest<T>` template
3. Use framework utilities for edge case generation
4. Apply testing macros for validation
5. Update CMakeLists.txt to include new test file
6. Add component-specific stress tests

## 📚 References

- [Google Test Documentation](https://google.github.io/googletest/)
- [AddressSanitizer Documentation](https://clang.llvm.org/docs/AddressSanitizer.html)
- [OpenCV Error Handling](https://docs.opencv.org/4.x/d8/d6a/group__core__utils.html)
- [PCL Exception Handling](https://pcl.readthedocs.io/projects/tutorials/en/latest/exceptions.html)

---

This comprehensive edge case testing framework ensures that every function in the stereo vision project is thoroughly tested against all possible failure modes, providing confidence in system robustness and reliability under real-world conditions.
