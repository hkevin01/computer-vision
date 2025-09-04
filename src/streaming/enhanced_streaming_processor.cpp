#include "streaming/enhanced_streaming_processor.hpp"
#include <QTimer>
#include <QDebug>

namespace cv_stereo {
namespace streaming {

EnhancedStreamingProcessor::EnhancedStreamingProcessor(const StreamingConfig& config, QObject* parent)
    : LiveStereoProcessor(parent), config_(config) {

    // Initialize timers for monitoring
    stats_timer_ = new QTimer(this);
    connect(stats_timer_, &QTimer::timeout, this, &EnhancedStreamingProcessor::update_streaming_stats);
    stats_timer_->start(1000);  // Update stats every second

    // Connect to base class signals to intercept processing
    connect(this, &LiveStereoProcessor::disparityMapReady,
            this, &EnhancedStreamingProcessor::process_streaming_frame);

    auto& logger = StructuredLogger::instance();
    logger.log_info("Enhanced streaming processor initialized", {
        {"target_fps", std::to_string(config_.target_fps)},
        {"max_buffer_size", std::to_string(config_.max_input_buffer_size)},
        {"adaptive_quality", config_.enable_adaptive_quality ? "enabled" : "disabled"}
    });
}

EnhancedStreamingProcessor::~EnhancedStreamingProcessor() {
    auto& logger = StructuredLogger::instance();
    logger.log_info("Enhanced streaming processor destroyed");
}

bool EnhancedStreamingProcessor::push_stereo_frame_pair(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        return false;
    }

    auto current_time = std::chrono::high_resolution_clock::now();

    std::unique_lock<std::mutex> lock(input_mutex_);

    // Clean up old frames first
    cleanup_old_frames();

    // Check if we should drop this frame
    FrameData frame_data;
    frame_data.left_frame = left.clone();
    frame_data.right_frame = right.clone();
    frame_data.timestamp = current_time;
    frame_data.sequence_number = sequence_counter_++;

    if (should_drop_frame(frame_data)) {
        stats_.frames_dropped_input++;
        return false;
    }

    // Add to buffer
    input_buffer_.push(frame_data);

    // Limit buffer size
    while (input_buffer_.size() > config_.max_input_buffer_size) {
        input_buffer_.pop();
        stats_.frames_dropped_input++;
    }

    lock.unlock();

    // Process the frame using the base class
    cv::Size processing_size = get_current_processing_size();
    cv::Mat resized_left = resize_for_processing(left, processing_size);
    cv::Mat resized_right = resize_for_processing(right, processing_size);

    // Call base class processing
    LiveStereoProcessor::processFramePair(resized_left, resized_right);

    return true;
}

bool EnhancedStreamingProcessor::get_latest_disparity_with_timeout(cv::Mat& disparity, std::chrono::milliseconds timeout) {
    auto start_time = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start_time < timeout) {
        {
            std::lock_guard<std::mutex> lock(output_mutex_);
            if (!output_buffer_.empty()) {
                const auto& result = output_buffer_.back();
                if (!result.disparity_map.empty()) {
                    disparity = result.disparity_map.clone();
                    return true;
                }
            }
        }

        // Also try the base class method
        cv::Mat base_disparity = LiveStereoProcessor::getLastDisparityMap();
        if (!base_disparity.empty()) {
            disparity = base_disparity;
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return false;
}

bool EnhancedStreamingProcessor::get_latest_point_cloud_with_timeout(cv::Mat& point_cloud, std::chrono::milliseconds timeout) {
    auto start_time = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start_time < timeout) {
        {
            std::lock_guard<std::mutex> lock(output_mutex_);
            if (!output_buffer_.empty()) {
                const auto& result = output_buffer_.back();
                if (!result.point_cloud.empty()) {
                    point_cloud = result.point_cloud.clone();
                    return true;
                }
            }
        }

        // Also try the base class method
        cv::Mat base_point_cloud = LiveStereoProcessor::getLastPointCloud();
        if (!base_point_cloud.empty()) {
            point_cloud = base_point_cloud;
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return false;
}

void EnhancedStreamingProcessor::cleanup_old_frames() {
    auto current_time = std::chrono::high_resolution_clock::now();

    while (!input_buffer_.empty()) {
        const auto& front_frame = input_buffer_.front();
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - front_frame.timestamp);

        if (age > config_.max_frame_age) {
            input_buffer_.pop();
            stats_.frames_dropped_input++;
        } else {
            break;
        }
    }
}

bool EnhancedStreamingProcessor::should_drop_frame(const FrameData& frame_data) const {
    if (!config_.enable_frame_dropping) {
        return false;
    }

    // Drop if buffer is too full
    if (input_buffer_.size() >= config_.max_input_buffer_size * 0.8) {
        return true;
    }

    // Drop if frame is too old
    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - frame_data.timestamp);

    return age > config_.max_frame_age;
}

cv::Size EnhancedStreamingProcessor::get_current_processing_size() const {
    if (current_quality_index_ < config_.quality_levels.size()) {
        return config_.quality_levels[current_quality_index_];
    }
    return cv::Size(640, 480);  // Default fallback
}

cv::Mat EnhancedStreamingProcessor::resize_for_processing(const cv::Mat& input, const cv::Size& target_size) const {
    if (input.size() == target_size) {
        return input;
    }

    cv::Mat resized;
    cv::resize(input, resized, target_size, 0, 0, cv::INTER_LINEAR);
    return resized;
}

void EnhancedStreamingProcessor::process_streaming_frame() {
    // This is called when base class emits disparityMapReady
    // We can add our own processing here

    auto current_time = std::chrono::high_resolution_clock::now();

    ProcessingResult result;
    result.disparity_map = LiveStereoProcessor::getLastDisparityMap();
    result.point_cloud = LiveStereoProcessor::getLastPointCloud();
    result.timestamp = current_time;

    // Calculate processing time based on last input frame
    {
        std::lock_guard<std::mutex> input_lock(input_mutex_);
        if (!input_buffer_.empty()) {
            auto input_time = input_buffer_.back().timestamp;
            result.processing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - input_time).count();
        }
    }

    // Add to output buffer
    {
        std::lock_guard<std::mutex> output_lock(output_mutex_);
        output_buffer_.push(result);

        // Limit output buffer size
        while (output_buffer_.size() > config_.max_output_buffer_size) {
            output_buffer_.pop();
            stats_.frames_dropped_output++;
        }
    }

    // Update performance metrics
    update_performance_metrics(result.processing_time_ms);
}

void EnhancedStreamingProcessor::update_streaming_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    auto current_time = std::chrono::high_resolution_clock::now();
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

    // Update stats
    stats_.current_fps = static_cast<double>(frame_timestamps_.size());
    stats_.input_buffer_size = input_buffer_.size();
    stats_.output_buffer_size = output_buffer_.size();
    stats_.current_processing_size = get_current_processing_size();

    // Check if we need to adapt quality
    if (config_.enable_adaptive_quality) {
        adapt_quality_if_needed();
    }

    emit streaming_stats_updated(stats_);
}

void EnhancedStreamingProcessor::adapt_quality_if_needed() {
    auto current_time = std::chrono::high_resolution_clock::now();

    // Only adapt every few seconds to avoid thrashing
    if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_adaptation_time_).count() < 3) {
        return;
    }

    double processing_load = stats_.average_processing_time_ms / (1000.0 / config_.target_fps);

    if (processing_load > 0.8 && current_quality_index_ > 0) {
        // Decrease quality
        current_quality_index_--;
        last_adaptation_time_ = current_time;

        auto& logger = StructuredLogger::instance();
        logger.log_info("Quality adapted down", {
            {"new_level", std::to_string(current_quality_index_)},
            {"processing_load", std::to_string(processing_load)}
        });

        emit quality_level_adapted(get_current_processing_size());

    } else if (processing_load < 0.5 && current_quality_index_ < config_.quality_levels.size() - 1) {
        // Increase quality
        current_quality_index_++;
        last_adaptation_time_ = current_time;

        auto& logger = StructuredLogger::instance();
        logger.log_info("Quality adapted up", {
            {"new_level", std::to_string(current_quality_index_)},
            {"processing_load", std::to_string(processing_load)}
        });

        emit quality_level_adapted(get_current_processing_size());
    }
}

void EnhancedStreamingProcessor::update_performance_metrics(double processing_time_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_frames_processed++;

    // Update rolling average
    stats_.average_processing_time_ms =
        (stats_.average_processing_time_ms * (stats_.total_frames_processed - 1) + processing_time_ms)
        / stats_.total_frames_processed;
}

EnhancedStreamingProcessor::StreamingStats EnhancedStreamingProcessor::get_streaming_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void EnhancedStreamingProcessor::update_target_fps(double fps) {
    config_.target_fps = fps;

    auto& logger = StructuredLogger::instance();
    logger.log_info("Target FPS updated", {{"new_fps", std::to_string(fps)}});
}

void EnhancedStreamingProcessor::set_adaptive_quality_enabled(bool enabled) {
    config_.enable_adaptive_quality = enabled;

    auto& logger = StructuredLogger::instance();
    logger.log_info("Adaptive quality setting changed", {{"enabled", enabled ? "true" : "false"}});
}

void EnhancedStreamingProcessor::set_current_quality_level(const cv::Size& size) {
    // Find the closest quality level
    for (size_t i = 0; i < config_.quality_levels.size(); ++i) {
        if (config_.quality_levels[i].width == size.width &&
            config_.quality_levels[i].height == size.height) {
            current_quality_index_ = i;
            break;
        }
    }

    auto& logger = StructuredLogger::instance();
    logger.log_info("Quality level set manually", {
        {"width", std::to_string(size.width)},
        {"height", std::to_string(size.height)}
    });
}

// StreamingProcessorFactory Implementation
StreamingProcessorFactory::HardwareCapabilities StreamingProcessorFactory::detect_capabilities() {
    HardwareCapabilities caps;

    caps.cpu_cores = std::thread::hardware_concurrency();
    caps.opencv_version = CV_VERSION_MAJOR + CV_VERSION_MINOR * 0.1;

    // TODO: Add CUDA/HIP detection when available

    auto& logger = StructuredLogger::instance();
    logger.log_info("Hardware capabilities detected", {
        {"cpu_cores", std::to_string(caps.cpu_cores)},
        {"opencv_version", std::to_string(caps.opencv_version)}
    });

    return caps;
}

EnhancedStreamingProcessor::StreamingConfig StreamingProcessorFactory::create_optimal_config(const HardwareCapabilities& caps) {
    EnhancedStreamingProcessor::StreamingConfig config;

    if (caps.cpu_cores >= 8) {
        // High-end system
        config.target_fps = 60.0;
        config.max_input_buffer_size = 15;
        config.max_output_buffer_size = 8;
    } else if (caps.cpu_cores >= 4) {
        // Mid-range system
        config.target_fps = 30.0;
        config.max_input_buffer_size = 10;
        config.max_output_buffer_size = 5;
    } else {
        // Low-end system
        config.target_fps = 15.0;
        config.max_input_buffer_size = 6;
        config.max_output_buffer_size = 3;
    }

    return config;
}

std::unique_ptr<EnhancedStreamingProcessor> StreamingProcessorFactory::create_processor(const HardwareCapabilities& caps) {
    auto config = create_optimal_config(caps);
    return std::make_unique<EnhancedStreamingProcessor>(config);
}

EnhancedStreamingProcessor::StreamingConfig StreamingProcessorFactory::get_realtime_config() {
    EnhancedStreamingProcessor::StreamingConfig config;
    config.target_fps = 60.0;
    config.max_input_buffer_size = 5;  // Small buffer for low latency
    config.max_output_buffer_size = 2;
    config.enable_adaptive_quality = true;
    config.quality_levels = {{320, 240}, {480, 360}, {640, 480}};  // Focus on speed
    return config;
}

EnhancedStreamingProcessor::StreamingConfig StreamingProcessorFactory::get_quality_config() {
    EnhancedStreamingProcessor::StreamingConfig config;
    config.target_fps = 15.0;  // Lower FPS for higher quality
    config.max_input_buffer_size = 15;
    config.max_output_buffer_size = 10;
    config.enable_adaptive_quality = false;  // Fixed high quality
    config.quality_levels = {{640, 480}, {1280, 720}, {1920, 1080}};  // Focus on quality
    return config;
}

EnhancedStreamingProcessor::StreamingConfig StreamingProcessorFactory::get_balanced_config() {
    EnhancedStreamingProcessor::StreamingConfig config;
    config.target_fps = 30.0;
    config.max_input_buffer_size = 10;
    config.max_output_buffer_size = 5;
    config.enable_adaptive_quality = true;
    config.quality_levels = {{320, 240}, {640, 480}, {1280, 720}};  // Balanced options
    return config;
}

} // namespace streaming
} // namespace cv_stereo
