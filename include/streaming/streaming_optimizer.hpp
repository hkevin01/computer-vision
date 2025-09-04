#pragma once

#include "live_stereo_processor.hpp"
#include "logging/structured_logger.hpp"
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>

namespace cv_stereo {
namespace streaming {

/**
 * @brief Enhanced streaming processor that optimizes LiveStereoProcessor with advanced buffering
 *
 * This class provides:
 * - Frame buffering with intelligent dropping
 * - Adaptive frame rate control
 * - Performance monitoring
 * - Asynchronous processing pipeline
 */
class StreamingOptimizer {
public:
    struct StreamingConfig {
        size_t max_buffer_size = 10;
        std::chrono::milliseconds max_frame_age{500};
        bool enable_frame_dropping = true;
        bool enable_adaptive_fps = true;
        double target_fps = 30.0;
        double min_fps = 10.0;
        double max_fps = 60.0;
    };

    struct StreamingStats {
        double current_fps = 0.0;
        double average_processing_time_ms = 0.0;
        size_t buffer_size = 0;
        size_t frames_dropped = 0;
        size_t total_frames_processed = 0;
        std::chrono::milliseconds last_processing_time{0};
    };

    StreamingOptimizer(stereo_vision::LiveStereoProcessor* processor,
                       const StreamingConfig& config);
    ~StreamingOptimizer();

    // Core interface
    bool push_frame_pair(const cv::Mat& left, const cv::Mat& right);
    StreamingStats get_stats() const;
    void update_config(const StreamingConfig& config);

    // Control
    void start();
    void stop();
    bool is_running() const { return running_; }

private:
    struct FrameData {
        cv::Mat left_frame;
        cv::Mat right_frame;
        std::chrono::steady_clock::time_point timestamp;
        uint64_t sequence_number = 0;
    };

    // Processing methods
    void processing_loop();
    void cleanup_old_frames();
    bool should_drop_frame() const;
    void update_performance_metrics(std::chrono::milliseconds processing_time);
    void adapt_frame_rate_if_needed();

    // Core components
    stereo_vision::LiveStereoProcessor* processor_;
    StreamingConfig config_;

    // Threading
    std::atomic<bool> running_{false};
    std::thread processing_thread_;

    // Buffering
    std::queue<FrameData> frame_buffer_;
    mutable std::mutex buffer_mutex_;
    std::condition_variable buffer_condition_;

    // Performance tracking
    mutable std::mutex stats_mutex_;
    StreamingStats stats_;
    std::deque<std::chrono::steady_clock::time_point> frame_timestamps_;
    uint64_t sequence_counter_ = 0;

    // Adaptive control
    std::chrono::steady_clock::time_point last_adaptation_time_;
    std::deque<std::chrono::milliseconds> processing_times_;
};

/**
 * @brief Factory for creating streaming optimizers with different configurations
 */
class StreamingOptimizerFactory {
public:
    static StreamingOptimizer::StreamingConfig create_realtime_config();
    static StreamingOptimizer::StreamingConfig create_quality_config();
    static StreamingOptimizer::StreamingConfig create_balanced_config();

    static std::unique_ptr<StreamingOptimizer> create_realtime_optimizer(
        stereo_vision::LiveStereoProcessor* processor);
    static std::unique_ptr<StreamingOptimizer> create_quality_optimizer(
        stereo_vision::LiveStereoProcessor* processor);
    static std::unique_ptr<StreamingOptimizer> create_balanced_optimizer(
        stereo_vision::LiveStereoProcessor* processor);
};

} // namespace streaming
} // namespace cv_stereo
