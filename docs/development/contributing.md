# Contributing to Computer Vision Stereo Processing Library

Thank you for your interest in contributing to the Computer Vision Stereo Processing Library! This guide will help you get started with contributing code, documentation, or reporting issues.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our Code of Conduct.

## How to Contribute

There are many ways to contribute to this project:

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Describe the behavior you observed** and **what behavior you expected**
- **Include code samples** and configuration files if relevant
- **Provide system information** (OS, compiler, GPU, etc.)

#### Bug Report Template

```markdown
## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
A clear description of what actually happened.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Compiler: [e.g., GCC 11.2]
- OpenCV Version: [e.g., 4.8.0]
- GPU: [e.g., NVIDIA RTX 3070]
- Library Version: [e.g., v2.1.0]

## Additional Context
Add any other context about the problem here, including:
- Configuration files
- Sample images
- Log output
- Stack traces
```

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **A clear and descriptive title**
- **A detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to most users
- **List similar features** in other libraries if applicable

### 📝 Contributing Documentation

Documentation improvements are always appreciated:

- Fix typos, grammar, or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Add translations (future feature)

### 🔧 Contributing Code

#### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

    ```bash
    git clone https://github.com/your-username/computer-vision.git
    cd computer-vision
    ```

3. **Set up the development environment**:

    ```bash
    # Install dependencies
    ./scripts/setup_dev_environment.sh

    # Configure development build
    mkdir build-dev && cd build-dev
    cmake -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DBUILD_TESTS=ON \
          -DBUILD_EXAMPLES=ON \
          -DENABLE_COVERAGE=ON \
          ..

    # Build
    make -j$(nproc)
    ```

4. **Run tests** to ensure everything works:

    ```bash
    make test
    # Or with detailed output
    ctest --output-on-failure
    ```

#### Development Workflow

1. **Create a feature branch**:

    ```bash
    git checkout -b feature/your-feature-name
    ```

2. **Make your changes** following our coding standards

3. **Write tests** for your changes

4. **Update documentation** if needed

5. **Run the full test suite**:

    ```bash
    # Unit tests
    make test

    # Integration tests
    ./scripts/run_integration_tests.sh

    # Performance benchmarks
    ./build/benchmark_app
    ```

6. **Check code quality**:

    ```bash
    # Format code
    ./scripts/format_code.sh

    # Static analysis
    ./scripts/run_static_analysis.sh

    # Check for memory leaks
    ./scripts/check_memory_leaks.sh
    ```

7. **Commit your changes**:

    ```bash
    git add .
    git commit -m "Add feature: brief description

    - Detailed description of changes
    - Any breaking changes
    - Fixes #issue-number"
    ```

8. **Push to your fork**:

    ```bash
    git push origin feature/your-feature-name
    ```

9. **Create a Pull Request** on GitHub

## Coding Standards

### C++ Style Guide

We follow a modified Google C++ Style Guide with these key points:

#### Naming Conventions

```cpp
// Classes: PascalCase
class StereoProcessor {};
class CalibrationManager {};

// Functions and methods: snake_case
void process_frames();
bool initialize_cameras();

// Variables: snake_case
int max_disparity = 96;
std::string config_file = "settings.yaml";

// Constants: UPPER_CASE with underscores
const int MAX_CAMERAS = 8;
const double DEFAULT_BASELINE = 120.0;

// Namespaces: snake_case
namespace stereo_vision {
namespace streaming {
    // ...
}
}

// Files: snake_case
// stereo_processor.hpp
// calibration_manager.cpp
```

#### Code Formatting

```cpp
// Header guards: Use #pragma once
#pragma once

// Include order:
// 1. Related header
// 2. System headers
// 3. Third-party headers
// 4. Project headers
#include "stereo_processor.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "calibration/calibration_manager.hpp"

// Class layout
class StereoProcessor {
public:
    // Public types first
    enum class ProcessingMode {
        FAST,
        BALANCED,
        ACCURATE
    };

    // Constructors
    StereoProcessor();
    explicit StereoProcessor(const StereoConfig& config);

    // Destructor
    virtual ~StereoProcessor() = default;

    // Public methods
    bool process_frames(const cv::Mat& left, const cv::Mat& right);
    cv::Mat get_disparity() const;

private:
    // Private methods
    void initialize_algorithms();
    bool validate_input(const cv::Mat& image) const;

    // Private members
    StereoConfig config_;
    cv::Ptr<cv::StereoBM> bm_matcher_;
    cv::Ptr<cv::StereoSGBM> sgbm_matcher_;
};
```

#### Error Handling

```cpp
// Use exceptions for exceptional circumstances
class StereoVisionException : public std::runtime_error {
public:
    explicit StereoVisionException(const std::string& message)
        : std::runtime_error(message) {}
};

// Use return codes for expected failures
enum class ProcessingResult {
    SUCCESS,
    INVALID_INPUT,
    CALIBRATION_REQUIRED,
    PROCESSING_FAILED
};

// Validate inputs
ProcessingResult StereoProcessor::process_frames(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        return ProcessingResult::INVALID_INPUT;
    }

    if (!is_calibrated()) {
        return ProcessingResult::CALIBRATION_REQUIRED;
    }

    try {
        // Processing logic here
        return ProcessingResult::SUCCESS;
    } catch (const cv::Exception& e) {
        spdlog::error("OpenCV error: {}", e.what());
        return ProcessingResult::PROCESSING_FAILED;
    }
}
```

#### Memory Management

```cpp
// Prefer RAII and smart pointers
class ResourceManager {
public:
    ResourceManager() : buffer_(std::make_unique<float[]>(buffer_size_)) {}

private:
    static constexpr size_t buffer_size_ = 1024 * 1024;
    std::unique_ptr<float[]> buffer_;
};

// Use move semantics when appropriate
std::unique_ptr<StereoProcessor> create_processor(StereoConfig config) {
    return std::make_unique<StereoProcessor>(std::move(config));
}

// Avoid raw pointers except for optional parameters
void process_with_callback(const cv::Mat& image,
                          std::function<void(const cv::Mat&)> callback = nullptr) {
    // Process image
    if (callback) {
        callback(result);
    }
}
```

### Documentation Standards

#### Header Documentation

```cpp
/**
 * @file stereo_processor.hpp
 * @brief Core stereo vision processing functionality
 * @author Your Name
 * @date 2024-01-15
 */

/**
 * @brief Main class for stereo image processing and disparity computation
 *
 * The StereoProcessor class provides a high-level interface for stereo vision
 * processing, including disparity map generation, depth estimation, and 3D
 * point cloud creation.
 *
 * @example
 * @code
 * StereoProcessor processor;
 * processor.set_calibration(calibration_data);
 *
 * cv::Mat left = cv::imread("left.jpg");
 * cv::Mat right = cv::imread("right.jpg");
 *
 * if (processor.process_frames(left, right)) {
 *     cv::Mat disparity = processor.get_disparity();
 *     // Use disparity map...
 * }
 * @endcode
 */
class StereoProcessor {
public:
    /**
     * @brief Process a stereo image pair to compute disparity
     *
     * @param left Left camera image (grayscale or color)
     * @param right Right camera image (grayscale or color)
     * @return ProcessingResult Success or specific error code
     *
     * @pre Both images must have the same size and type
     * @pre Processor must be calibrated (see set_calibration())
     *
     * @post On success, disparity map is available via get_disparity()
     * @post Processing statistics are updated
     *
     * @throws StereoVisionException If critical error occurs during processing
     */
    ProcessingResult process_frames(const cv::Mat& left, const cv::Mat& right);
};
```

#### Inline Documentation

```cpp
// Use clear, concise comments for complex logic
void StereoProcessor::optimize_parameters() {
    // Adjust block size based on image resolution
    // Smaller images need smaller blocks for adequate detail
    if (image_size_.area() < 640 * 480) {
        config_.block_size = std::max(3, config_.block_size - 2);
    }

    // Scale disparity range with baseline and resolution
    // Closer cameras (smaller baseline) need larger disparity range
    const double baseline_factor = 120.0 / calibration_.baseline;  // mm
    config_.max_disparity = static_cast<int>(config_.max_disparity * baseline_factor);
}
```

### Testing Standards

#### Unit Tests

```cpp
#include <gtest/gtest.h>
#include "stereo_processor.hpp"

class StereoProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        processor_ = std::make_unique<StereoProcessor>();

        // Create test images
        left_image_ = cv::Mat::zeros(480, 640, CV_8UC1);
        right_image_ = cv::Mat::zeros(480, 640, CV_8UC1);

        // Add test pattern
        cv::rectangle(left_image_, cv::Rect(100, 100, 200, 200), 255, -1);
        cv::rectangle(right_image_, cv::Rect(110, 100, 200, 200), 255, -1);  // 10px disparity
    }

    std::unique_ptr<StereoProcessor> processor_;
    cv::Mat left_image_, right_image_;
};

TEST_F(StereoProcessorTest, ProcessValidImages) {
    // Load test calibration
    CalibrationData calibration = load_test_calibration();
    processor_->set_calibration(calibration);

    // Process test images
    auto result = processor_->process_frames(left_image_, right_image_);
    EXPECT_EQ(result, ProcessingResult::SUCCESS);

    // Verify disparity output
    cv::Mat disparity = processor_->get_disparity();
    EXPECT_FALSE(disparity.empty());
    EXPECT_EQ(disparity.type(), CV_16S);
}

TEST_F(StereoProcessorTest, RejectEmptyImages) {
    cv::Mat empty_image;
    auto result = processor_->process_frames(empty_image, right_image_);
    EXPECT_EQ(result, ProcessingResult::INVALID_INPUT);
}

TEST_F(StereoProcessorTest, RequireCalibration) {
    // Don't set calibration
    auto result = processor_->process_frames(left_image_, right_image_);
    EXPECT_EQ(result, ProcessingResult::CALIBRATION_REQUIRED);
}
```

#### Integration Tests

```cpp
// Test entire processing pipeline
TEST(IntegrationTest, FullStereoProcessingPipeline) {
    // Load real calibration data
    CalibrationData calibration;
    ASSERT_TRUE(load_calibration_from_file("test_data/calibration.yaml", calibration));

    // Load test image pair
    cv::Mat left = cv::imread("test_data/stereo_left.jpg");
    cv::Mat right = cv::imread("test_data/stereo_right.jpg");
    ASSERT_FALSE(left.empty() && right.empty());

    // Create and configure processor
    StereoProcessor processor;
    processor.set_calibration(calibration);

    StereoConfig config;
    config.algorithm = StereoAlgorithm::SGBM;
    config.max_disparity = 96;
    processor.set_config(config);

    // Process images
    auto result = processor.process_frames(left, right);
    ASSERT_EQ(result, ProcessingResult::SUCCESS);

    // Verify disparity quality
    cv::Mat disparity = processor.get_disparity();

    // Check coverage (should have > 50% valid disparities)
    cv::Mat valid_mask = disparity > 0;
    double coverage = cv::sum(valid_mask)[0] / (disparity.rows * disparity.cols * 255.0);
    EXPECT_GT(coverage, 0.5);

    // Check disparity range
    double min_disp, max_disp;
    cv::minMaxLoc(disparity, &min_disp, &max_disp, nullptr, nullptr, valid_mask);
    EXPECT_GE(min_disp, 0);
    EXPECT_LE(max_disp, config.max_disparity);
}
```

#### Performance Tests

```cpp
// Benchmark processing performance
TEST(PerformanceTest, ProcessingSpeed) {
    StereoProcessor processor;

    // Setup with real calibration and config
    setup_processor_for_performance(processor);

    cv::Mat left = create_test_image(1920, 1080);
    cv::Mat right = create_test_image(1920, 1080);

    // Warm up
    processor.process_frames(left, right);

    // Benchmark
    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        processor.process_frames(left, right);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double avg_time_ms = duration.count() / static_cast<double>(iterations);
    double fps = 1000.0 / avg_time_ms;

    std::cout << "Average processing time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Estimated FPS: " << fps << std::endl;

    // Performance requirements (adjust based on target hardware)
    EXPECT_LT(avg_time_ms, 100.0);  // < 100ms for 1080p
    EXPECT_GT(fps, 10.0);           // > 10 FPS minimum
}
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:

    ```bash
    make test
    ./scripts/run_integration_tests.sh
    ```

2. **Run code quality checks**:

    ```bash
    ./scripts/format_code.sh
    ./scripts/run_static_analysis.sh
    ```

3. **Update documentation** if your changes affect the API

4. **Add tests** for new functionality

5. **Check performance impact** with benchmarks

### Pull Request Template

Use this template for your pull request description:

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Performance Impact
Describe any performance implications of your changes.

## Breaking Changes
List any breaking changes and migration steps required.

## Related Issues
Fixes #issue-number
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** on multiple platforms if needed
4. **Documentation review** for user-facing changes
5. **Performance review** for performance-critical changes

### After Approval

- Your PR will be merged using "Squash and merge" to maintain a clean history
- The feature branch will be automatically deleted
- Release notes will be updated if necessary

## Development Tools and Scripts

### Useful Scripts

```bash
# Development environment setup
./scripts/setup_dev_environment.sh

# Code formatting (uses clang-format)
./scripts/format_code.sh

# Static analysis (uses clang-tidy, cppcheck)
./scripts/run_static_analysis.sh

# Memory leak detection (uses valgrind)
./scripts/check_memory_leaks.sh

# Performance profiling
./scripts/profile_performance.sh

# Documentation generation
./scripts/build_docs.sh

# Clean build artifacts
./scripts/clean_all.sh
```

### Development Configuration

#### VS Code Settings

If you use VS Code, these settings are recommended:

```json
{
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "C_Cpp.default.cppStandard": "c++17",
    "files.associations": {
        "*.hpp": "cpp"
    },
    "cmake.buildDirectory": "${workspaceFolder}/build-dev",
    "cmake.generator": "Unix Makefiles",
    "cmake.configureArgs": [
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DBUILD_TESTS=ON"
    ]
}
```

#### Git Hooks

Set up pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
./scripts/install_git_hooks.sh

# This will run formatting and basic checks before each commit
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update version numbers in CMakeLists.txt
2. Update CHANGELOG.md with release notes
3. Run full test suite on all supported platforms
4. Create release tag: `git tag -a v2.1.0 -m "Release v2.1.0"`
5. Push tag: `git push origin v2.1.0`
6. GitHub Actions will automatically build and publish releases

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussion
- **Stack Overflow**: Technical questions (tag: `computer-vision-stereo`)

### Maintainer Response Times

- **Critical bugs**: Within 24 hours
- **Bug reports**: Within 1 week
- **Feature requests**: Within 2 weeks
- **Pull requests**: Within 1 week

Thank you for contributing to the Computer Vision Stereo Processing Library! 🎉

---

!!! info "Questions?"
    If you have any questions about contributing, feel free to open a discussion on GitHub or reach out to the maintainers.

!!! tip "First Time Contributing?"
    Look for issues labeled "good first issue" or "help wanted" - these are great starting points for new contributors.
