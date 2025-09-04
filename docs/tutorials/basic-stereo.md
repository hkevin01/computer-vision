# Basic Stereo Processing Tutorial

This tutorial will guide you through the fundamentals of stereo vision processing using the Computer Vision Stereo Processing Library. You'll learn how to set up a basic stereo processing pipeline from camera calibration to 3D point cloud generation.

## Prerequisites

Before starting this tutorial, ensure you have:

- The Computer Vision Stereo Processing Library installed
- Two USB cameras or a stereo camera setup
- Sample stereo image pairs (provided in `data/sample_images/`)
- Basic familiarity with computer vision concepts

## Tutorial Overview

This tutorial covers:

1. **Camera calibration** - Setting up your stereo camera system
2. **Basic stereo processing** - Computing disparity maps
3. **Depth map generation** - Converting disparity to real-world depth
4. **Point cloud creation** - Generating 3D point clouds
5. **Optimization** - Improving processing performance

## Step 1: Camera Calibration

### Understanding Calibration

Camera calibration is essential for accurate stereo vision. It determines:

- **Intrinsic parameters**: Focal length, principal point, lens distortion
- **Extrinsic parameters**: Relative position and orientation between cameras
- **Rectification parameters**: Matrices to align image pairs

### Preparing Calibration Images

You'll need 15-20 stereo image pairs of a calibration pattern:

```bash
# Use the calibration data generator
./build/generate_test_data calibration

# Or capture your own images
mkdir calibration_images
./build/live_stereo_tuning --mode calibration --output calibration_images/
```

For best results:

- Use a high-quality checkerboard pattern (9x6 or 8x6 squares)
- Ensure good lighting and sharp images
- Vary the pattern position, angle, and distance
- Cover the entire field of view
- Keep the cameras perfectly synchronized

### Running Calibration

#### Method 1: Using the GUI Tool

```bash
# Launch the live calibration wizard
./build/live_stereo_tuning

# Follow the on-screen instructions:
# 1. Select calibration mode
# 2. Position checkerboard pattern
# 3. Capture images when prompted
# 4. Review calibration quality
# 5. Save calibration file
```

#### Method 2: Using Code

```cpp
#include "stereo_vision_core.hpp"

int main() {
    // Create calibration manager
    stereo_vision::CalibrationManager calibrator;

    // Configure calibration settings
    stereo_vision::CalibrationConfig config;
    config.board_size = cv::Size(9, 6);  // 9x6 checkerboard
    config.square_size = 25.0;           // 25mm squares
    calibrator.setConfig(config);

    // Add calibration images
    for (int i = 1; i <= 20; ++i) {
        std::string left_file = fmt::format("calibration/left_{:02d}.jpg", i);
        std::string right_file = fmt::format("calibration/right_{:02d}.jpg", i);

        cv::Mat left = cv::imread(left_file);
        cv::Mat right = cv::imread(right_file);

        if (calibrator.addCalibrationImage(left, right)) {
            std::cout << "Added calibration image " << i << std::endl;
        } else {
            std::cout << "Pattern not found in image " << i << std::endl;
        }
    }

    // Perform calibration
    if (calibrator.calibrate()) {
        // Save calibration results
        calibrator.saveCalibration("stereo_calibration.yaml");

        // Check calibration quality
        auto quality = calibrator.getCalibrationQuality();
        std::cout << "Calibration RMS error: " << calibrator.getReprojectionError() << std::endl;
        std::cout << "Calibration quality: " << quality.overall_score << std::endl;

    } else {
        std::cerr << "Calibration failed!" << std::endl;
        return -1;
    }

    return 0;
}
```

### Evaluating Calibration Quality

Good calibration should have:

- **RMS reprojection error < 1.0 pixels** (preferably < 0.5)
- **Even distribution** of calibration points
- **Stable results** across multiple calibration runs

```cpp
// Check calibration quality metrics
auto quality = calibrator.getCalibrationQuality();
std::cout << "Overall score: " << quality.overall_score << std::endl;
std::cout << "RMS error: " << quality.rms_error << std::endl;
std::cout << "Coverage score: " << quality.coverage_score << std::endl;
std::cout << "Stability score: " << quality.stability_score << std::endl;
```

## Step 2: Basic Stereo Processing

### Setting Up the Processor

```cpp
#include "stereo_vision_core.hpp"

int main() {
    // Create stereo processor
    stereo_vision::StereoProcessor processor;

    // Load calibration
    stereo_vision::CalibrationData calibration;
    if (!loadCalibrationFromFile("stereo_calibration.yaml", calibration)) {
        std::cerr << "Failed to load calibration!" << std::endl;
        return -1;
    }
    processor.setCalibration(calibration);

    // Configure stereo processing
    stereo_vision::StereoConfig config;
    config.algorithm = stereo_vision::StereoAlgorithm::SGBM;
    config.block_size = 5;
    config.min_disparity = 0;
    config.max_disparity = 96;
    config.enable_post_processing = true;
    processor.setConfig(config);

    return 0;
}
```

### Processing Your First Stereo Pair

```cpp
// Load stereo image pair
cv::Mat left_img = cv::imread("data/sample_images/left.jpg");
cv::Mat right_img = cv::imread("data/sample_images/right.jpg");

if (left_img.empty() || right_img.empty()) {
    std::cerr << "Could not load images!" << std::endl;
    return -1;
}

// Process the stereo pair
if (processor.processFrames(left_img, right_img)) {
    // Get disparity map
    cv::Mat disparity = processor.getDisparity();

    // Normalize disparity for display
    cv::Mat disparity_display;
    cv::normalize(disparity, disparity_display, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply color map for better visualization
    cv::Mat disparity_color;
    cv::applyColorMap(disparity_display, disparity_color, cv::COLORMAP_JET);

    // Display results
    cv::imshow("Left Image", left_img);
    cv::imshow("Right Image", right_img);
    cv::imshow("Disparity Map", disparity_color);
    cv::waitKey(0);

    // Save results
    cv::imwrite("disparity_result.png", disparity_display);
    cv::imwrite("disparity_color.png", disparity_color);

} else {
    std::cerr << "Stereo processing failed!" << std::endl;
}
```

### Understanding Disparity Maps

Disparity maps show the pixel difference between corresponding points in stereo images:

- **Bright areas**: Close objects (high disparity)
- **Dark areas**: Far objects (low disparity)
- **Black pixels**: No correspondence found (invalid)

The disparity value `d` relates to depth `Z` by:
```
Z = (f * B) / d
```
Where:
- `f` = focal length (pixels)
- `B` = baseline (distance between cameras)
- `d` = disparity (pixels)

## Step 3: Depth Map Generation

Convert disparity to real-world depth measurements:

```cpp
// Get depth map in millimeters
cv::Mat depth = processor.getDepthMap();

if (!depth.empty()) {
    // Find depth statistics
    double min_depth, max_depth;
    cv::minMaxLoc(depth, &min_depth, &max_depth);

    std::cout << "Depth range: " << min_depth << " - " << max_depth << " mm" << std::endl;

    // Create visualization
    cv::Mat depth_display;
    cv::normalize(depth, depth_display, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat depth_color;
    cv::applyColorMap(depth_display, depth_color, cv::COLORMAP_PLASMA);

    cv::imshow("Depth Map", depth_color);

    // Save depth data
    cv::FileStorage fs("depth_data.yml", cv::FileStorage::WRITE);
    fs << "depth_map" << depth;
    fs << "min_depth" << min_depth;
    fs << "max_depth" << max_depth;
    fs.release();

    // Query depth at specific points
    cv::Point center(depth.cols/2, depth.rows/2);
    float center_depth = depth.at<float>(center.y, center.x);
    std::cout << "Depth at center: " << center_depth << " mm" << std::endl;
}
```

## Step 4: Point Cloud Generation

Create 3D point clouds from depth data:

```cpp
// Generate point cloud with color
auto point_cloud = processor.getPointCloud();

if (point_cloud && !point_cloud->empty()) {
    std::cout << "Generated point cloud with " << point_cloud->size() << " points" << std::endl;

    // Save as PLY file
    pcl::io::savePLYFile("output.ply", *point_cloud);

    // Save as PCD file (PCL format)
    pcl::io::savePCDFile("output.pcd", *point_cloud);

    // Basic point cloud statistics
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*point_cloud, min_pt, max_pt);

    std::cout << "Point cloud bounds:" << std::endl;
    std::cout << "  X: " << min_pt.x << " to " << max_pt.x << std::endl;
    std::cout << "  Y: " << min_pt.y << " to " << max_pt.y << std::endl;
    std::cout << "  Z: " << min_pt.z << " to " << max_pt.z << std::endl;

    // Filter point cloud (remove noise)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Statistical outlier removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(point_cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*filtered_cloud);

    pcl::io::savePLYFile("output_filtered.ply", *filtered_cloud);
    std::cout << "Filtered cloud has " << filtered_cloud->size() << " points" << std::endl;
}
```

### Visualizing Point Clouds

```cpp
// Optional: Visualize point cloud (requires PCL visualization)
#ifdef PCL_VISUALIZATION_AVAILABLE
pcl::visualization::CloudViewer viewer("Point Cloud Viewer");
viewer.showCloud(point_cloud);

// Keep viewer open
while (!viewer.wasStopped()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
#endif
```

## Step 5: Optimization and Performance

### Tuning Parameters for Your Setup

Different stereo setups require different parameters:

```cpp
// For close-range objects (< 2 meters)
config.max_disparity = 128;
config.block_size = 3;
config.sgbm_p1 = 8 * 3 * config.block_size * config.block_size;
config.sgbm_p2 = 32 * 3 * config.block_size * config.block_size;

// For distant objects (> 5 meters)
config.max_disparity = 64;
config.block_size = 7;
config.enable_post_processing = true;

// For real-time processing
config.algorithm = stereo_vision::StereoAlgorithm::BM;
config.enable_gpu = true;
config.num_threads = std::thread::hardware_concurrency();
```

### Performance Monitoring

```cpp
// Monitor processing performance
auto stats = processor.getStats();
std::cout << "Processing time: " << stats.processing_time_ms << " ms" << std::endl;
std::cout << "FPS: " << stats.fps << std::endl;
std::cout << "Disparity coverage: " << stats.disparity_coverage * 100 << "%" << std::endl;
std::cout << "Memory usage: " << stats.memory_usage_mb << " MB" << std::endl;
```

### Advanced Optimization with StreamingOptimizer

For real-time applications, use the StreamingOptimizer:

```cpp
#include "streaming/streaming_optimizer.hpp"

// Create streaming configuration
stereo_vision::streaming::StreamingConfig stream_config;
stream_config.buffer_size = 10;
stream_config.max_fps = 30.0;
stream_config.adaptive_quality = true;
stream_config.worker_threads = 4;

// Create streaming optimizer
auto optimizer = stereo_vision::streaming::StreamingOptimizerFactory::create(stream_config);
optimizer->start();

// Process frames in real-time
cv::Mat left_frame, right_frame;
while (capture_frames(left_frame, right_frame)) {
    optimizer->processFrame(left_frame, right_frame);

    // Get latest result (non-blocking)
    if (auto result = optimizer->getLatestResult()) {
        cv::imshow("Live Disparity", result->disparity);
    }

    cv::waitKey(1);
}

optimizer->stop();
```

## Complete Example Application

Here's a complete example that combines all the concepts:

```cpp
#include "stereo_vision_core.hpp"
#include "streaming/streaming_optimizer.hpp"
#include <iostream>
#include <fmt/format.h>

int main(int argc, char* argv[]) {
    try {
        // 1. Load calibration
        stereo_vision::CalibrationData calibration;
        if (!loadCalibrationFromFile("stereo_calibration.yaml", calibration)) {
            std::cerr << "Failed to load calibration file" << std::endl;
            return -1;
        }

        // 2. Configure processor
        stereo_vision::StereoConfig config;
        config.algorithm = stereo_vision::StereoAlgorithm::SGBM;
        config.block_size = 5;
        config.max_disparity = 96;
        config.enable_post_processing = true;
        config.enable_gpu = true;

        // 3. Create processor
        stereo_vision::StereoProcessor processor(config);
        processor.setCalibration(calibration);

        // 4. Process sample images
        cv::Mat left = cv::imread("data/sample_images/left.jpg");
        cv::Mat right = cv::imread("data/sample_images/right.jpg");

        if (left.empty() || right.empty()) {
            std::cerr << "Could not load sample images" << std::endl;
            return -1;
        }

        std::cout << "Processing stereo pair..." << std::endl;

        if (processor.processFrames(left, right)) {
            // 5. Get results
            cv::Mat disparity = processor.getDisparity();
            cv::Mat depth = processor.getDepthMap();
            auto point_cloud = processor.getPointCloud();

            // 6. Save results
            cv::Mat disparity_display;
            cv::normalize(disparity, disparity_display, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite("tutorial_disparity.png", disparity_display);

            if (point_cloud) {
                pcl::io::savePLYFile("tutorial_pointcloud.ply", *point_cloud);
                std::cout << "Point cloud saved with " << point_cloud->size() << " points" << std::endl;
            }

            // 7. Display statistics
            auto stats = processor.getStats();
            std::cout << fmt::format("Processing completed in {:.2f} ms", stats.processing_time_ms) << std::endl;
            std::cout << fmt::format("Disparity coverage: {:.1f}%", stats.disparity_coverage * 100) << std::endl;

            // 8. Show results
            cv::Mat disparity_color;
            cv::applyColorMap(disparity_display, disparity_color, cv::COLORMAP_JET);

            cv::imshow("Original Left", left);
            cv::imshow("Disparity Map", disparity_color);
            cv::waitKey(0);

            std::cout << "Tutorial completed successfully!" << std::endl;

        } else {
            std::cerr << "Stereo processing failed" << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
```

### Building and Running the Example

```bash
# Compile the example
g++ -std=c++17 tutorial_example.cpp \
    -I/path/to/computer-vision/include \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d \
    -lpcl_common -lpcl_io -lpcl_filters \
    -lstereo_vision_core -lfmt \
    -o tutorial_example

# Run the example
./tutorial_example
```

## Troubleshooting Common Issues

### Poor Disparity Quality

**Problem**: Sparse or noisy disparity maps

**Solutions**:
1. Improve calibration with more images
2. Adjust block size (try 3, 5, 7, or 9)
3. Tune disparity range (min/max disparity)
4. Enable post-processing filters
5. Check lighting conditions

```cpp
// Better parameters for noisy environments
config.block_size = 7;  // Larger block for stability
config.enable_post_processing = true;
config.enable_speckle_filtering = true;
config.sgbm_speckle_window_size = 150;
config.sgbm_speckle_range = 16;
```

### Slow Processing Performance

**Problem**: Low frame rates or high processing times

**Solutions**:
1. Enable GPU acceleration
2. Reduce image resolution
3. Lower disparity range
4. Use faster algorithm (BM instead of SGBM)
5. Reduce block size

```cpp
// Performance-optimized configuration
config.algorithm = stereo_vision::StereoAlgorithm::BM;
config.block_size = 5;
config.max_disparity = 64;
config.enable_gpu = true;
config.num_threads = std::thread::hardware_concurrency();
```

### Calibration Issues

**Problem**: High reprojection error or unstable calibration

**Solutions**:
1. Use more calibration images (20+)
2. Improve lighting and image sharpness
3. Cover entire field of view with pattern
4. Use larger checkerboard pattern
5. Ensure cameras are rigidly mounted

## Next Steps

After completing this tutorial, you can:

1. **Explore advanced features**: Try AI-enhanced stereo matching
2. **Real-time processing**: Set up live camera streaming
3. **Multi-camera systems**: Process multiple stereo pairs
4. **Custom algorithms**: Implement your own stereo matching
5. **Integration**: Use the library in your own applications

### Recommended Learning Path

1. **[Streaming Optimization Guide](../user-guide/streaming.md)**: Learn real-time processing
2. **[Multi-camera Tutorial](multi-camera.md)**: Work with multiple cameras
3. **[AI Enhancement Tutorial](ai-stereo.md)**: Use neural stereo matching
4. **[Custom Algorithm Tutorial](custom-stereo.md)**: Implement custom processing

---

!!! success "Congratulations!"
    You've completed the basic stereo processing tutorial! You now understand the fundamentals of stereo vision and can process your own stereo images.

!!! tip "Performance"
    For real-time applications, make sure to check out the [Streaming Optimization Guide](../user-guide/streaming.md) to achieve the best performance.

!!! info "Community"
    Share your results and get help from the community in our [GitHub Discussions](https://github.com/computer-vision-project/computer-vision/discussions).
