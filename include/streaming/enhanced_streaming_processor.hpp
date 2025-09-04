#pragma once

#include <memory>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "gui/live_stereo_processor.hpp"
#include "logging/structured_logger.hpp"

namespace cv_stereo {
namespace streaming {

/**
 * @brief Enhanced streaming pipeline that extends LiveStereoProcessor with advanced buffering
 */
class EnhancedStreamingProcessor : public LiveStereoProcessor {

public:
    struct StreamingConfig {
        // Buffer configuration
        size_t max_input_buffer_size = 10;
        size_t max_output_buffer_size = 5;
        bool enable_frame_dropping = true;
        std::chrono::milliseconds max_frame_age{500};

        // Processing configuration
        bool enable_adaptive_quality = true;
        bool enable_gpu_upload_async = true;
        bool enable_result_caching = false;

        // Performance targets
        double target_fps = 30.0;
        double max_processing_latency_ms = 100.0;

        // Quality levels for adaptive processing
        std::vector<cv::Size> quality_levels = {{320, 240}, {640, 480}, {1280, 720}};
    };

    explicit EnhancedStreamingProcessor(const StreamingConfig& config = StreamingConfig{}, QObject* parent = nullptr);
    ~EnhancedStreamingProcessor();

    // Enhanced interface
    bool push_stereo_frame_pair(const cv::Mat& left, const cv::Mat& right);
    bool get_latest_disparity_with_timeout(cv::Mat& disparity, std::chrono::milliseconds timeout = std::chrono::milliseconds(50));
    bool get_latest_point_cloud_with_timeout(cv::Mat& point_cloud, std::chrono::milliseconds timeout = std::chrono::milliseconds(50));

    // Performance monitoring
    struct StreamingStats {
        double current_fps = 0.0;
        double average_processing_time_ms = 0.0;
        size_t input_buffer_size = 0;
        size_t output_buffer_size = 0;
        size_t frames_dropped_input = 0;
        size_t frames_dropped_output = 0;
        size_t total_frames_processed = 0;
        cv::Size current_processing_size{640, 480};

        // GPU-specific stats
        bool gpu_processing_available = false;
        double gpu_utilization_percent = 0.0;
    };

    StreamingStats get_streaming_stats() const;

    // Configuration updates
    void update_target_fps(double fps);
    void set_adaptive_quality_enabled(bool enabled);
    void set_current_quality_level(const cv::Size& size);

signals:
    void streaming_stats_updated(const StreamingStats& stats);
    void quality_level_adapted(const cv::Size& new_size);

private slots:
    void process_streaming_frame();
    void update_streaming_stats();

private:
    struct FrameData {
        cv::Mat left_frame;
        cv::Mat right_frame;
        std::chrono::high_resolution_clock::time_point timestamp;
        uint64_t sequence_number;
    };

    struct ProcessingResult {
        cv::Mat disparity_map;
        cv::Mat point_cloud;
        std::chrono::high_resolution_clock::time_point timestamp;
        double processing_time_ms;
    };

    StreamingConfig config_;

    // Enhanced buffering
    std::queue<FrameData> input_buffer_;
    std::queue<ProcessingResult> output_buffer_;
    mutable std::mutex input_mutex_;
    mutable std::mutex output_mutex_;

    // Statistics and monitoring
    mutable std::mutex stats_mutex_;
    StreamingStats stats_;
    std::atomic<uint64_t> sequence_counter_{0};
    std::deque<std::chrono::high_resolution_clock::time_point> frame_timestamps_;

    // Adaptive quality control
    std::atomic<size_t> current_quality_index_{1};  // Start with medium quality
    std::chrono::high_resolution_clock::time_point last_adaptation_time_;
    QTimer* stats_timer_;

    // Private methods
    void cleanup_old_frames();
    void adapt_quality_if_needed();
    bool should_drop_frame(const FrameData& frame_data) const;
    cv::Size get_current_processing_size() const;
    cv::Mat resize_for_processing(const cv::Mat& input, const cv::Size& target_size) const;
    void update_performance_metrics(double processing_time_ms);
};

/**
 * @brief Factory for creating optimized streaming processors
 */
class StreamingProcessorFactory {
public:
    struct HardwareCapabilities {
        bool has_cuda = false;
        bool has_hip = false;
        size_t gpu_memory_mb = 0;
        size_t cpu_cores = 0;
        double opencv_version = 0.0;
    };

    static HardwareCapabilities detect_capabilities();
    static EnhancedStreamingProcessor::StreamingConfig create_optimal_config(const HardwareCapabilities& caps);
    static std::unique_ptr<EnhancedStreamingProcessor> create_processor(const HardwareCapabilities& caps);

    // Preset configurations
    static EnhancedStreamingProcessor::StreamingConfig get_realtime_config();
    static EnhancedStreamingProcessor::StreamingConfig get_quality_config();
    static EnhancedStreamingProcessor::StreamingConfig get_balanced_config();
};

} // namespace streaming
} // namespace cv_stereo
