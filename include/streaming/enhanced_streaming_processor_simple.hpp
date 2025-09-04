#pragma once

#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <deque>
#include <opencv2/opencv.hpp>
#include "gui/live_stereo_processor.hpp"
#include "logging/structured_logger.hpp"

namespace cv_stereo {
namespace streaming {

/**
 * @brief Enhanced streaming processor with advanced buffering and performance optimization
 *
 * This class extends the existing LiveStereoProcessor to provide:
 * - Frame buffering with intelligent dropping
 * - Adaptive quality based on performance
 * - Timeout-based retrieval of results
 * - Performance monitoring and statistics
 */
class EnhancedStreamingProcessor {
public:
    struct StreamingConfig {
        // Buffer configuration
        size_t max_input_buffer_size = 10;
        size_t max_output_buffer_size = 5;
        bool enable_frame_dropping = true;
        std::chrono::milliseconds max_frame_age{500};

        // Processing configuration
        bool enable_adaptive_quality = true;
        bool enable_result_caching = false;

        // Performance targets
        double target_fps = 30.0;
        double max_processing_latency_ms = 100.0;

        // Quality levels for adaptive processing
        std::vector<cv::Size> quality_levels = {{320, 240}, {640, 480}, {1280, 720}};
    };

    struct StreamingStats {
        double current_fps = 0.0;
        double average_processing_time_ms = 0.0;
        size_t input_buffer_size = 0;
        size_t output_buffer_size = 0;
        size_t frames_dropped_input = 0;
        size_t frames_dropped_output = 0;
        size_t total_frames_processed = 0;
        cv::Size current_processing_size{640, 480};
    };

    explicit EnhancedStreamingProcessor(const StreamingConfig& config = StreamingConfig{});
    ~EnhancedStreamingProcessor();

    // Core interface
    bool push_stereo_frame_pair(const cv::Mat& left, const cv::Mat& right);
    bool get_latest_disparity_with_timeout(cv::Mat& disparity, std::chrono::milliseconds timeout = std::chrono::milliseconds(50));
    bool get_latest_point_cloud_with_timeout(cv::Mat& point_cloud, std::chrono::milliseconds timeout = std::chrono::milliseconds(50));

    // Configuration and monitoring
    StreamingStats get_streaming_stats() const;
    void update_target_fps(double fps);
    void set_adaptive_quality_enabled(bool enabled);
    void set_current_quality_level(const cv::Size& size);

private:
    struct FrameData {
        cv::Mat left_frame;
        cv::Mat right_frame;
        std::chrono::high_resolution_clock::time_point timestamp;
        uint64_t sequence_number = 0;
    };

    struct ProcessingResult {
        cv::Mat disparity_map;
        cv::Mat point_cloud;
        std::chrono::high_resolution_clock::time_point timestamp;
        double processing_time_ms = 0.0;
    };

    // Core processing
    void process_frame_pair(const cv::Mat& left, const cv::Mat& right);
    void cleanup_old_frames();
    bool should_drop_frame(const FrameData& frame_data) const;

    // Quality management
    cv::Size get_current_processing_size() const;
    cv::Mat resize_for_processing(const cv::Mat& input, const cv::Size& target_size) const;
    void adapt_quality_if_needed();

    // Performance monitoring
    void update_performance_metrics(double processing_time_ms);
    void update_streaming_stats();

    // Configuration
    StreamingConfig config_;

    // Buffering
    std::queue<FrameData> input_buffer_;
    std::queue<ProcessingResult> output_buffer_;
    mutable std::mutex input_mutex_;
    mutable std::mutex output_mutex_;

    // Processing state
    std::unique_ptr<LiveStereoProcessor> processor_;
    size_t current_quality_index_ = 1;  // Start with medium quality
    uint64_t sequence_counter_ = 0;

    // Performance tracking
    mutable std::mutex stats_mutex_;
    StreamingStats stats_;
    std::deque<std::chrono::high_resolution_clock::time_point> frame_timestamps_;
    std::chrono::high_resolution_clock::time_point last_adaptation_time_;

    // Threading
    std::atomic<bool> processing_enabled_{true};
    std::thread processing_thread_;
    void processing_loop();
};

/**
 * @brief Factory for creating optimized streaming processors based on hardware capabilities
 */
class StreamingProcessorFactory {
public:
    struct HardwareCapabilities {
        size_t cpu_cores = 1;
        bool has_cuda = false;
        bool has_opencl = false;
        double opencv_version = 4.0;
        size_t available_memory_mb = 1024;
    };

    static HardwareCapabilities detect_capabilities();
    static EnhancedStreamingProcessor::StreamingConfig create_optimal_config(const HardwareCapabilities& caps);
    static std::unique_ptr<EnhancedStreamingProcessor> create_processor(const HardwareCapabilities& caps);

    // Predefined configurations
    static EnhancedStreamingProcessor::StreamingConfig get_realtime_config();
    static EnhancedStreamingProcessor::StreamingConfig get_quality_config();
    static EnhancedStreamingProcessor::StreamingConfig get_balanced_config();
};

} // namespace streaming
} // namespace cv_stereo
