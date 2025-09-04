# Streaming Optimization Guide

The Computer Vision Stereo Processing Library includes advanced streaming optimization features designed for real-time applications. This guide covers how to configure and use the `StreamingOptimizer` for maximum performance.

## Overview

The `StreamingOptimizer` is a high-performance component that enhances the standard `LiveStereoProcessor` with:

- **Intelligent buffering** with adaptive frame dropping
- **Multi-threaded processing** with configurable worker threads
- **Performance monitoring** and adaptive quality control
- **GPU stream overlap** for CUDA/HIP acceleration
- **Automatic FPS adjustment** based on system load

## Basic Usage

### Quick Start

```cpp
#include "streaming/streaming_optimizer.hpp"

// Create streaming configuration
stereo_vision::streaming::StreamingConfig config;
config.buffer_size = 10;
config.max_fps = 30;
config.adaptive_quality = true;
config.worker_threads = 4;

// Create and start optimizer
auto optimizer = stereo_vision::streaming::StreamingOptimizerFactory::create(config);
optimizer->start();

// Process frames
cv::Mat left_frame, right_frame;
while (capture_frames(left_frame, right_frame)) {
    optimizer->processFrame(left_frame, right_frame);

    // Get results (non-blocking)
    if (auto result = optimizer->getLatestResult()) {
        cv::imshow("Disparity", result->disparity);
        cv::imshow("Point Cloud Preview", result->preview);
    }

    // Monitor performance
    auto stats = optimizer->getStats();
    std::cout << "FPS: " << stats.current_fps
              << ", Dropped: " << stats.frames_dropped << std::endl;
}

optimizer->stop();
```

### Integration with Existing Code

The optimizer extends `LiveStereoProcessor`, so it's a drop-in replacement:

```cpp
// Before - standard processor
std::unique_ptr<stereo_vision::LiveStereoProcessor> processor =
    std::make_unique<stereo_vision::LiveStereoProcessor>();

// After - optimized streaming
auto config = stereo_vision::streaming::StreamingConfig{};
config.adaptive_quality = true;
std::unique_ptr<stereo_vision::LiveStereoProcessor> processor =
    stereo_vision::streaming::StreamingOptimizerFactory::create(config);
```

## Configuration Options

### StreamingConfig Parameters

```cpp
struct StreamingConfig {
    size_t buffer_size = 5;           // Frame buffer size
    double max_fps = 30.0;            // Target maximum FPS
    size_t worker_threads = 4;        // Processing thread count
    bool adaptive_quality = true;     // Enable adaptive quality
    bool enable_gpu_overlap = true;   // GPU stream overlap
    double quality_threshold = 0.8;   // Quality vs speed balance
    size_t max_queue_size = 20;       // Maximum queue size
    bool enable_frame_dropping = true; // Drop frames when overloaded
};
```

### Detailed Configuration

#### Buffer Management

```cpp
config.buffer_size = 10;              // Larger buffer = smoother playback
config.max_queue_size = 20;           // Maximum frames in processing queue
config.enable_frame_dropping = true;  // Drop old frames when buffer full
```

**Buffer size guidelines:**
- **Small (3-5)**: Low latency, may drop frames under load
- **Medium (5-10)**: Balanced latency and smoothness
- **Large (10+)**: Smooth playback, higher latency

#### Threading Configuration

```cpp
config.worker_threads = std::thread::hardware_concurrency(); // Use all cores
config.worker_threads = 4;  // Fixed thread count
config.worker_threads = 1;  // Single-threaded processing
```

**Thread count recommendations:**
- **Single camera pair**: 2-4 threads
- **Multiple cameras**: 4-8 threads
- **High resolution**: 6-12 threads
- **GPU processing**: 2-4 threads (CPU threads for I/O)

#### Adaptive Quality

```cpp
config.adaptive_quality = true;
config.quality_threshold = 0.8;  // 80% target performance
config.max_fps = 30.0;           // Never exceed this FPS
```

When adaptive quality is enabled:
- Automatically reduces processing quality under load
- Adjusts disparity range and block size
- Disables expensive post-processing when needed
- Maintains target frame rate

#### GPU Acceleration

```cpp
config.enable_gpu_overlap = true;   // Enable CUDA/HIP streams
```

GPU overlap features:
- Parallel CPU and GPU processing
- Asynchronous memory transfers
- Multiple GPU streams for pipeline stages
- Automatic fallback to CPU if GPU unavailable

## Performance Monitoring

### Real-time Statistics

```cpp
auto stats = optimizer->getStats();

std::cout << "Current FPS: " << stats.current_fps << std::endl;
std::cout << "Average FPS: " << stats.average_fps << std::endl;
std::cout << "Frames processed: " << stats.frames_processed << std::endl;
std::cout << "Frames dropped: " << stats.frames_dropped << std::endl;
std::cout << "Processing time: " << stats.average_processing_time_ms << "ms" << std::endl;
std::cout << "Queue size: " << stats.current_queue_size << std::endl;
std::cout << "Buffer utilization: " << stats.buffer_utilization << "%" << std::endl;
```

### Performance Metrics

The `StreamingStats` structure provides comprehensive metrics:

```cpp
struct StreamingStats {
    double current_fps;                    // Current frame rate
    double average_fps;                    // Average frame rate
    uint64_t frames_processed;             // Total frames processed
    uint64_t frames_dropped;               // Total frames dropped
    double average_processing_time_ms;     // Average processing time
    size_t current_queue_size;             // Current queue size
    double buffer_utilization;             // Buffer usage percentage
    std::chrono::steady_clock::time_point last_update; // Last update time
};
```

### Performance Logging

Enable detailed performance logging:

```cpp
// Set log level to see performance metrics
spdlog::set_level(spdlog::level::info);

// The optimizer automatically logs:
// - Frame processing times
// - Queue sizes and buffer utilization
// - Adaptive quality adjustments
// - Frame drop events
```

## Optimization Strategies

### For Low Latency

```cpp
StreamingConfig config;
config.buffer_size = 3;              // Minimal buffering
config.max_fps = 60.0;               // High frame rate
config.adaptive_quality = true;      // Reduce quality under load
config.enable_frame_dropping = true; // Drop old frames aggressively
config.worker_threads = 2;           // Minimal threading overhead
```

### For High Quality

```cpp
StreamingConfig config;
config.buffer_size = 10;             // Larger buffer for smoothness
config.max_fps = 24.0;               // Cinematic frame rate
config.adaptive_quality = false;     // Maintain quality
config.enable_frame_dropping = false; // Process all frames
config.worker_threads = 8;           // More processing power
```

### For Multiple Cameras

```cpp
// Create separate optimizers for each camera pair
std::vector<std::unique_ptr<StreamingOptimizer>> optimizers;

for (int camera_id = 0; camera_id < num_cameras; ++camera_id) {
    StreamingConfig config;
    config.buffer_size = 5;
    config.max_fps = 30.0 / num_cameras;  // Divide frame rate
    config.worker_threads = 2;            // Fewer threads per camera

    optimizers.push_back(StreamingOptimizerFactory::create(config));
}
```

### For GPU Processing

```cpp
StreamingConfig config;
config.enable_gpu_overlap = true;    // Enable GPU streams
config.worker_threads = 4;           // CPU threads for I/O
config.buffer_size = 8;              // Larger buffer for GPU pipeline
config.adaptive_quality = true;      // Adapt to GPU load
```

## Advanced Features

### Custom Processing Callbacks

```cpp
optimizer->setProcessingCallback([](const cv::Mat& left, const cv::Mat& right) {
    // Custom pre-processing
    cv::Mat left_enhanced, right_enhanced;
    enhance_contrast(left, left_enhanced);
    enhance_contrast(right, right_enhanced);
    return std::make_pair(left_enhanced, right_enhanced);
});

optimizer->setResultCallback([](const StereoResult& result) {
    // Custom post-processing
    save_to_database(result);
    update_visualization(result);
});
```

### Dynamic Configuration

```cpp
// Adjust configuration at runtime
optimizer->updateConfig([](StreamingConfig& config) {
    config.max_fps = new_target_fps;
    config.adaptive_quality = enable_adaptation;
});

// Monitor system load and adjust
if (system_load > 0.8) {
    optimizer->updateConfig([](StreamingConfig& config) {
        config.quality_threshold = 0.6;  // Reduce quality
        config.max_fps = 20.0;           // Lower frame rate
    });
}
```

### Frame Synchronization

```cpp
// For multi-camera synchronization
class SynchronizedStreaming {
    std::vector<std::unique_ptr<StreamingOptimizer>> optimizers_;
    std::mutex sync_mutex_;

public:
    void processFrames(const std::vector<StereoFramePair>& frames) {
        std::lock_guard<std::mutex> lock(sync_mutex_);

        for (size_t i = 0; i < optimizers_.size(); ++i) {
            optimizers_[i]->processFrame(frames[i].left, frames[i].right);
        }
    }
};
```

## Troubleshooting

### Common Performance Issues

#### High Frame Drop Rate

```cpp
// Symptoms: stats.frames_dropped increasing rapidly
// Solutions:
config.buffer_size += 2;           // Increase buffer
config.worker_threads += 2;        // More processing power
config.adaptive_quality = true;    // Enable quality adaptation
config.max_fps *= 0.8;            // Reduce target FPS
```

#### High Latency

```cpp
// Symptoms: Noticeable delay between input and output
// Solutions:
config.buffer_size = std::max(3, config.buffer_size - 2); // Reduce buffer
config.enable_frame_dropping = true;  // Drop old frames
config.max_fps += 5;                   // Allow higher FPS
```

#### Inconsistent Performance

```cpp
// Symptoms: FPS varies widely
// Solutions:
config.adaptive_quality = true;    // Enable adaptation
config.worker_threads = std::thread::hardware_concurrency(); // Use all cores
// Check system background processes
// Ensure adequate cooling for sustained performance
```

### GPU Issues

#### CUDA/HIP Not Available

```cpp
// The optimizer automatically falls back to CPU processing
// Check GPU availability:
bool gpu_available = optimizer->isGPUAvailable();
if (!gpu_available) {
    spdlog::warn("GPU acceleration not available, using CPU processing");
    config.enable_gpu_overlap = false;
}
```

#### GPU Memory Issues

```cpp
// Monitor GPU memory usage
auto gpu_stats = optimizer->getGPUStats();
if (gpu_stats.memory_usage > 0.9) {
    // Reduce processing load
    config.buffer_size = std::max(3, config.buffer_size - 2);
    config.max_fps *= 0.8;
}
```

### Debugging Tools

```cpp
// Enable verbose logging
spdlog::set_level(spdlog::level::debug);

// Monitor in real-time
auto monitor_thread = std::thread([&optimizer]() {
    while (optimizer->isRunning()) {
        auto stats = optimizer->getStats();
        std::cout << "FPS: " << stats.current_fps
                  << ", Queue: " << stats.current_queue_size
                  << ", Dropped: " << stats.frames_dropped << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
});
```

## Best Practices

### Configuration Guidelines

1. **Start with defaults** and adjust based on performance
2. **Monitor statistics** to identify bottlenecks
3. **Enable adaptive quality** for varying system loads
4. **Use appropriate buffer sizes** for your latency requirements
5. **Match thread count** to your CPU capabilities

### System Optimization

1. **CPU affinity**: Pin threads to specific cores
2. **Process priority**: Increase priority for real-time applications
3. **Memory allocation**: Pre-allocate buffers to avoid runtime allocation
4. **GPU scheduling**: Use dedicated GPU contexts for streaming

### Performance Testing

```cpp
// Benchmark different configurations
std::vector<StreamingConfig> test_configs = {
    {5, 30.0, 2, true, true, 0.8, 15, true},   // Low latency
    {10, 24.0, 4, false, true, 0.9, 20, false}, // High quality
    {8, 30.0, 6, true, true, 0.7, 25, true}    // Balanced
};

for (const auto& config : test_configs) {
    auto optimizer = StreamingOptimizerFactory::create(config);
    // Run benchmark...
    auto final_stats = run_benchmark(optimizer.get(), test_data);
    std::cout << "Config performance: " << final_stats.average_fps << " FPS" << std::endl;
}
```

---

!!! tip "Performance Tip"
    Start with adaptive quality enabled and monitor the statistics to understand your system's capabilities before fine-tuning the configuration.

!!! warning "Threading"
    More threads isn't always better - excessive threading can cause context switching overhead. Start with 2-4 threads and increase based on performance metrics.

!!! info "GPU Acceleration"
    GPU acceleration provides the biggest performance boost for high-resolution processing. Ensure your GPU drivers are up to date for optimal performance.
