#include "streaming/streaming_optimizer.hpp"
#include <algorithm>

namespace cv_stereo {
namespace streaming {

StreamingOptimizer::StreamingOptimizer(stereo_vision::LiveStereoProcessor* processor,
                                       const StreamingConfig& config)
    : processor_(processor), config_(config),
      last_adaptation_time_(std::chrono::steady_clock::now()) {

    if (!processor_) {
        throw std::invalid_argument("LiveStereoProcessor cannot be null");
    }

    auto& logger = StructuredLogger::instance();
    logger.log_info("StreamingOptimizer initialized", {
        {"target_fps", std::to_string(config_.target_fps)},
        {"max_buffer_size", std::to_string(config_.max_buffer_size)},
        {"adaptive_fps", config_.enable_adaptive_fps ? "enabled" : "disabled"}
    });
}

StreamingOptimizer::~StreamingOptimizer() {
    stop();

    auto& logger = StructuredLogger::instance();
    logger.log_info("StreamingOptimizer destroyed", {
        {"total_frames_processed", std::to_string(stats_.total_frames_processed)},
        {"frames_dropped", std::to_string(stats_.frames_dropped)}
    });
}

bool StreamingOptimizer::push_frame_pair(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        return false;
    }

    std::unique_lock<std::mutex> lock(buffer_mutex_);

    // Clean up old frames first
    cleanup_old_frames();

    // Check if we should drop this frame
    if (should_drop_frame()) {
        stats_.frames_dropped++;
        return false;
    }

    // Create frame data
    FrameData frame_data;
    frame_data.left_frame = left.clone();
    frame_data.right_frame = right.clone();
    frame_data.timestamp = std::chrono::steady_clock::now();
    frame_data.sequence_number = sequence_counter_++;

    // Add to buffer
    frame_buffer_.push(frame_data);

    // Update buffer size stats
    stats_.buffer_size = frame_buffer_.size();

    lock.unlock();
    buffer_condition_.notify_one();

    return true;
}

StreamingOptimizer::StreamingStats StreamingOptimizer::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void StreamingOptimizer::update_config(const StreamingConfig& config) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    config_ = config;

    auto& logger = StructuredLogger::instance();
    logger.log_info("StreamingOptimizer config updated", {
        {"new_target_fps", std::to_string(config_.target_fps)},
        {"new_buffer_size", std::to_string(config_.max_buffer_size)}
    });
}

void StreamingOptimizer::start() {
    if (running_) {
        return;
    }

    running_ = true;
    processing_thread_ = std::thread(&StreamingOptimizer::processing_loop, this);

    auto& logger = StructuredLogger::instance();
    logger.log_info("StreamingOptimizer started");
}

void StreamingOptimizer::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    buffer_condition_.notify_all();

    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }

    auto& logger = StructuredLogger::instance();
    logger.log_info("StreamingOptimizer stopped");
}

void StreamingOptimizer::processing_loop() {
    auto& logger = StructuredLogger::instance();
    logger.log_info("StreamingOptimizer processing loop started");

    while (running_) {
        std::unique_lock<std::mutex> lock(buffer_mutex_);

        // Wait for frames or stop signal
        buffer_condition_.wait(lock, [this] {
            return !running_ || !frame_buffer_.empty();
        });

        if (!running_) {
            break;
        }

        if (frame_buffer_.empty()) {
            continue;
        }

        // Get the next frame
        FrameData frame_data = frame_buffer_.front();
        frame_buffer_.pop();
        stats_.buffer_size = frame_buffer_.size();

        lock.unlock();

        // Process the frame
        auto start_time = std::chrono::steady_clock::now();

        processor_->processFramePair(frame_data.left_frame, frame_data.right_frame);

        auto end_time = std::chrono::steady_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        // Update performance metrics
        update_performance_metrics(processing_time);

        // Update frame rate statistics
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            auto current_time = std::chrono::steady_clock::now();
            frame_timestamps_.push_back(current_time);

            // Keep only recent timestamps (last second)
            while (!frame_timestamps_.empty()) {
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - frame_timestamps_.front());
                if (age > std::chrono::milliseconds(1000)) {
                    frame_timestamps_.pop_front();
                } else {
                    break;
                }
            }

            stats_.current_fps = static_cast<double>(frame_timestamps_.size());
            stats_.total_frames_processed++;
            stats_.last_processing_time = processing_time;
        }

        // Adapt frame rate if needed
        if (config_.enable_adaptive_fps) {
            adapt_frame_rate_if_needed();
        }
    }

    logger.log_info("StreamingOptimizer processing loop ended");
}

void StreamingOptimizer::cleanup_old_frames() {
    auto current_time = std::chrono::steady_clock::now();

    while (!frame_buffer_.empty()) {
        const auto& front_frame = frame_buffer_.front();
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - front_frame.timestamp);

        if (age > config_.max_frame_age) {
            frame_buffer_.pop();
            stats_.frames_dropped++;
        } else {
            break;
        }
    }
}

bool StreamingOptimizer::should_drop_frame() const {
    if (!config_.enable_frame_dropping) {
        return false;
    }

    // Drop if buffer is too full
    return frame_buffer_.size() >= config_.max_buffer_size;
}

void StreamingOptimizer::update_performance_metrics(std::chrono::milliseconds processing_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Update rolling average of processing times
    processing_times_.push_back(processing_time);

    // Keep only recent processing times (last 10 frames)
    if (processing_times_.size() > 10) {
        processing_times_.pop_front();
    }

    // Calculate average
    auto total_time = std::chrono::milliseconds(0);
    for (const auto& time : processing_times_) {
        total_time += time;
    }

    if (!processing_times_.empty()) {
        stats_.average_processing_time_ms =
            static_cast<double>(total_time.count()) / processing_times_.size();
    }
}

void StreamingOptimizer::adapt_frame_rate_if_needed() {
    auto current_time = std::chrono::steady_clock::now();

    // Only adapt every few seconds to avoid thrashing
    if (std::chrono::duration_cast<std::chrono::seconds>(
            current_time - last_adaptation_time_).count() < 3) {
        return;
    }

    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Calculate processing load
    double frame_interval_ms = 1000.0 / config_.target_fps;
    double processing_load = stats_.average_processing_time_ms / frame_interval_ms;

    auto& logger = StructuredLogger::instance();

    if (processing_load > 0.8 && config_.target_fps > config_.min_fps) {
        // Decrease target FPS to reduce load
        config_.target_fps = std::max(config_.min_fps, config_.target_fps * 0.8);
        last_adaptation_time_ = current_time;

        logger.log_info("Frame rate adapted down", {
            {"new_fps", std::to_string(config_.target_fps)},
            {"processing_load", std::to_string(processing_load)}
        });

    } else if (processing_load < 0.5 && config_.target_fps < config_.max_fps) {
        // Increase target FPS as we have headroom
        config_.target_fps = std::min(config_.max_fps, config_.target_fps * 1.2);
        last_adaptation_time_ = current_time;

        logger.log_info("Frame rate adapted up", {
            {"new_fps", std::to_string(config_.target_fps)},
            {"processing_load", std::to_string(processing_load)}
        });
    }
}

// StreamingOptimizerFactory Implementation

StreamingOptimizer::StreamingConfig StreamingOptimizerFactory::create_realtime_config() {
    StreamingOptimizer::StreamingConfig config;
    config.max_buffer_size = 3;  // Small buffer for low latency
    config.max_frame_age = std::chrono::milliseconds(100);  // Drop frames quickly
    config.enable_frame_dropping = true;
    config.enable_adaptive_fps = true;
    config.target_fps = 60.0;
    config.min_fps = 30.0;
    config.max_fps = 120.0;
    return config;
}

StreamingOptimizer::StreamingConfig StreamingOptimizerFactory::create_quality_config() {
    StreamingOptimizer::StreamingConfig config;
    config.max_buffer_size = 15;  // Large buffer for quality
    config.max_frame_age = std::chrono::milliseconds(1000);  // Keep frames longer
    config.enable_frame_dropping = false;  // Process all frames
    config.enable_adaptive_fps = false;  // Fixed frame rate
    config.target_fps = 15.0;
    config.min_fps = 5.0;
    config.max_fps = 30.0;
    return config;
}

StreamingOptimizer::StreamingConfig StreamingOptimizerFactory::create_balanced_config() {
    StreamingOptimizer::StreamingConfig config;
    config.max_buffer_size = 8;  // Medium buffer
    config.max_frame_age = std::chrono::milliseconds(500);  // Balanced aging
    config.enable_frame_dropping = true;
    config.enable_adaptive_fps = true;
    config.target_fps = 30.0;
    config.min_fps = 15.0;
    config.max_fps = 60.0;
    return config;
}

std::unique_ptr<StreamingOptimizer> StreamingOptimizerFactory::create_realtime_optimizer(
    stereo_vision::LiveStereoProcessor* processor) {
    return std::make_unique<StreamingOptimizer>(processor, create_realtime_config());
}

std::unique_ptr<StreamingOptimizer> StreamingOptimizerFactory::create_quality_optimizer(
    stereo_vision::LiveStereoProcessor* processor) {
    return std::make_unique<StreamingOptimizer>(processor, create_quality_config());
}

std::unique_ptr<StreamingOptimizer> StreamingOptimizerFactory::create_balanced_optimizer(
    stereo_vision::LiveStereoProcessor* processor) {
    return std::make_unique<StreamingOptimizer>(processor, create_balanced_config());
}

} // namespace streaming
} // namespace cv_stereo
