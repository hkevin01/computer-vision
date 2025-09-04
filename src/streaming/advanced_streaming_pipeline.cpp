#include "streaming/advanced_streaming_pipeline.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#endif

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace cv_stereo {
namespace streaming {

// AdvancedFrameBuffer Implementation
AdvancedFrameBuffer::AdvancedFrameBuffer(const BufferConfig& config)
    : config_(config) {
#ifdef USE_CUDA
    initialize_cuda_streams();
#endif

    auto& logger = StructuredLogger::instance();
    logger.log_info("Advanced frame buffer initialized", {
        {"max_buffer_size", config_.max_buffer_size},
        {"gpu_buffer_size", config_.gpu_buffer_size},
        {"enable_gpu_upload", config_.enable_gpu_upload}
    });
}

AdvancedFrameBuffer::~AdvancedFrameBuffer() {
#ifdef USE_CUDA
    cleanup_cuda_streams();
#endif
}

bool AdvancedFrameBuffer::push_frame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    // Check buffer size and drop old frames if necessary
    if (cpu_buffer_.size() >= config_.drop_threshold) {
        stats_.total_frames_dropped++;

        // Drop oldest frame
        if (!cpu_buffer_.empty()) {
            cpu_buffer_.pop();
        }

        auto& logger = StructuredLogger::instance();
        logger.log_warning("Frame dropped due to buffer overflow", {
            {"buffer_size", cpu_buffer_.size()},
            {"drop_threshold", config_.drop_threshold}
        });
    }

    // Clean up old frames
    cleanup_old_frames();

    // Create new frame data
    FrameData frame_data;
    frame_data.cpu_frame = frame.clone();
    frame_data.timestamp = std::chrono::high_resolution_clock::now();
    frame_data.sequence_number = sequence_counter_++;

    // Upload to GPU if enabled and available
    if (config_.enable_gpu_upload && gpu_buffer_.size() < config_.gpu_buffer_size) {
        if (upload_to_gpu(frame_data)) {
            gpu_buffer_.push(frame_data);
        }
    }

    cpu_buffer_.push(frame_data);
    stats_.total_frames_pushed++;

    // Update peak size
    if (cpu_buffer_.size() > stats_.peak_size) {
        stats_.peak_size = cpu_buffer_.size();
    }

    buffer_condition_.notify_one();
    return true;
}

bool AdvancedFrameBuffer::pop_frame(FrameData& frame_data) {
    std::unique_lock<std::mutex> lock(buffer_mutex_);

    if (cpu_buffer_.empty()) {
        return false;
    }

    frame_data = cpu_buffer_.front();
    cpu_buffer_.pop();

    return true;
}

bool AdvancedFrameBuffer::peek_latest_frame(FrameData& frame_data) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    if (cpu_buffer_.empty()) {
        return false;
    }

    frame_data = cpu_buffer_.back();
    return true;
}

size_t AdvancedFrameBuffer::size() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return cpu_buffer_.size();
}

void AdvancedFrameBuffer::cleanup_old_frames() {
    auto current_time = std::chrono::high_resolution_clock::now();

    while (!cpu_buffer_.empty()) {
        const auto& front_frame = cpu_buffer_.front();
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - front_frame.timestamp);

        if (age > config_.max_frame_age) {
            cpu_buffer_.pop();
            stats_.total_frames_dropped++;
        } else {
            break;
        }
    }
}

bool AdvancedFrameBuffer::upload_to_gpu(FrameData& frame_data) {
#ifdef USE_CUDA
    try {
        if (!upload_streams_.empty()) {
            cudaStream_t stream = upload_streams_[0];  // Use first stream for simplicity
            frame_data.gpu_frame.upload(frame_data.cpu_frame, stream);
            frame_data.cuda_stream = stream;
            return true;
        }
    } catch (const cv::Exception& e) {
        stats_.gpu_upload_failures++;
        auto& logger = StructuredLogger::instance();
        logger.log_error("GPU upload failed", {
            {"error", e.what()}
        });
    }
#endif
    return false;
}

#ifdef USE_CUDA
void AdvancedFrameBuffer::initialize_cuda_streams() {
    const int num_streams = 2;
    upload_streams_.resize(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&upload_streams_[i]);
    }
}

void AdvancedFrameBuffer::cleanup_cuda_streams() {
    for (auto stream : upload_streams_) {
        cudaStreamDestroy(stream);
    }
    upload_streams_.clear();
}
#endif

AdvancedFrameBuffer::BufferStats AdvancedFrameBuffer::get_stats() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    BufferStats stats = stats_;
    stats.current_size = cpu_buffer_.size();
    return stats;
}

// GPUStreamProcessor Implementation
GPUStreamProcessor::GPUStreamProcessor(const ProcessingConfig& config)
    : config_(config) {
    initialize_gpu_resources();

    auto& logger = StructuredLogger::instance();
    logger.log_info("GPU stream processor initialized", {
        {"num_cuda_streams", config_.num_cuda_streams},
        {"enable_stream_overlap", config_.enable_stream_overlap}
    });
}

GPUStreamProcessor::~GPUStreamProcessor() {
    cleanup_gpu_resources();
}

std::future<cv::Mat> GPUStreamProcessor::process_stereo_async(const cv::Mat& left, const cv::Mat& right) {
    return std::async(std::launch::async, [this, left, right]() {
        return process_stereo_sync(left, right);
    });
}

cv::Mat GPUStreamProcessor::process_stereo_sync(const cv::Mat& left, const cv::Mat& right) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check cache first
    if (config_.enable_result_caching) {
        auto hash_pair = compute_frame_hash(left, right);
        std::lock_guard<std::mutex> cache_lock(cache_mutex_);
        auto it = result_cache_.find(hash_pair);
        if (it != result_cache_.end()) {
            return it->second.clone();
        }
    }

    cv::Mat result;

#ifdef USE_CUDA
    if (!cuda_streams_.empty() && cuda_stereo_matcher_) {
        size_t stream_index = get_next_stream_index();
        result = process_on_stream(left, right, stream_index);
    } else
#endif
    {
        // CPU fallback
        if (!cpu_stereo_matcher_) {
            cpu_stereo_matcher_ = cv::StereoBM::create(config_.max_disparity, config_.block_size);
        }

        cv::Mat left_gray, right_gray;
        if (left.channels() == 3) {
            cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
        } else {
            left_gray = left;
            right_gray = right;
        }

        cpu_stereo_matcher_->compute(left_gray, right_gray, result);
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    {
        std::lock_guard<std::mutex> lock(processing_mutex_);
        stats_.total_processed_frames++;
        stats_.average_processing_time_ms =
            (stats_.average_processing_time_ms * (stats_.total_processed_frames - 1) +
             duration.count() / 1000.0) / stats_.total_processed_frames;
    }

    // Cache result if enabled
    if (config_.enable_result_caching && !result.empty()) {
        auto hash_pair = compute_frame_hash(left, right);
        std::lock_guard<std::mutex> cache_lock(cache_mutex_);
        result_cache_[hash_pair] = result.clone();

        // Limit cache size
        if (result_cache_.size() > 50) {
            result_cache_.erase(result_cache_.begin());
        }
    }

    return result;
}

bool GPUStreamProcessor::is_gpu_available() const {
#ifdef USE_CUDA
    return !cuda_streams_.empty();
#else
    return false;
#endif
}

void GPUStreamProcessor::initialize_gpu_resources() {
#ifdef USE_CUDA
    try {
        int device_count;
        cudaGetDeviceCount(&device_count);

        if (device_count > 0) {
            cuda_streams_.resize(config_.num_cuda_streams);
            for (int i = 0; i < config_.num_cuda_streams; ++i) {
                cudaStreamCreate(&cuda_streams_[i]);
            }

            cuda_stereo_matcher_ = cv::cuda::StereoBM::create(config_.max_disparity, config_.block_size);

            auto& logger = StructuredLogger::instance();
            logger.log_info("CUDA resources initialized", {
                {"device_count", device_count},
                {"streams_created", static_cast<int>(cuda_streams_.size())}
            });
        }
    } catch (const cv::Exception& e) {
        auto& logger = StructuredLogger::instance();
        logger.log_error("Failed to initialize CUDA resources", {
            {"error", e.what()}
        });
    }
#endif

    // Initialize CPU fallback
    cpu_stereo_matcher_ = cv::StereoBM::create(config_.max_disparity, config_.block_size);
}

void GPUStreamProcessor::cleanup_gpu_resources() {
#ifdef USE_CUDA
    for (auto stream : cuda_streams_) {
        cudaStreamDestroy(stream);
    }
    cuda_streams_.clear();
    cuda_stereo_matcher_.reset();
#endif
}

size_t GPUStreamProcessor::get_next_stream_index() {
    return current_stream_index_++ % cuda_streams_.size();
}

cv::Mat GPUStreamProcessor::process_on_stream(const cv::Mat& left, const cv::Mat& right, size_t stream_index) {
#ifdef USE_CUDA
    if (stream_index >= cuda_streams_.size()) {
        return cv::Mat();
    }

    cudaStream_t stream = cuda_streams_[stream_index];

    cv::cuda::GpuMat gpu_left, gpu_right, gpu_result;

    // Upload to GPU
    gpu_left.upload(left, stream);
    gpu_right.upload(right, stream);

    // Convert to grayscale if needed
    if (left.channels() == 3) {
        cv::cuda::GpuMat gpu_left_gray, gpu_right_gray;
        cv::cuda::cvtColor(gpu_left, gpu_left_gray, cv::COLOR_BGR2GRAY, 0, stream);
        cv::cuda::cvtColor(gpu_right, gpu_right_gray, cv::COLOR_BGR2GRAY, 0, stream);

        cuda_stereo_matcher_->compute(gpu_left_gray, gpu_right_gray, gpu_result, stream);
    } else {
        cuda_stereo_matcher_->compute(gpu_left, gpu_right, gpu_result, stream);
    }

    // Download result
    cv::Mat result;
    gpu_result.download(result, stream);

    // Synchronize stream
    cudaStreamSynchronize(stream);

    return result;
#else
    return cv::Mat();
#endif
}

std::pair<size_t, size_t> GPUStreamProcessor::compute_frame_hash(const cv::Mat& left, const cv::Mat& right) {
    // Simple hash based on image properties - could be improved
    size_t left_hash = left.rows ^ (left.cols << 1) ^ (left.type() << 2);
    size_t right_hash = right.rows ^ (right.cols << 1) ^ (right.type() << 2);

    if (!left.empty()) {
        left_hash ^= std::hash<uchar>{}(left.at<uchar>(left.rows/2, left.cols/2));
    }
    if (!right.empty()) {
        right_hash ^= std::hash<uchar>{}(right.at<uchar>(right.rows/2, right.cols/2));
    }

    return {left_hash, right_hash};
}

GPUStreamProcessor::ProcessingStats GPUStreamProcessor::get_stats() const {
    std::lock_guard<std::mutex> lock(processing_mutex_);
    return stats_;
}

// AdvancedStreamingPipeline Implementation
AdvancedStreamingPipeline::AdvancedStreamingPipeline(const PipelineConfig& config)
    : config_(config) {

    frame_buffer_ = std::make_unique<AdvancedFrameBuffer>(config_.buffer_config);
    gpu_processor_ = std::make_unique<GPUStreamProcessor>(config_.processing_config);

    auto& logger = StructuredLogger::instance();
    logger.log_info("Advanced streaming pipeline created", {
        {"target_fps", config_.target_fps},
        {"max_latency_ms", config_.max_latency_ms},
        {"enable_adaptive_quality", config_.enable_adaptive_quality}
    });
}

AdvancedStreamingPipeline::~AdvancedStreamingPipeline() {
    stop();
}

bool AdvancedStreamingPipeline::start() {
    if (is_running_.load()) {
        return false;
    }

    is_running_.store(true);

    processing_thread_ = std::thread(&AdvancedStreamingPipeline::processing_loop, this);
    monitoring_thread_ = std::thread(&AdvancedStreamingPipeline::monitoring_loop, this);

    auto& logger = StructuredLogger::instance();
    logger.log_info("Streaming pipeline started");

    return true;
}

void AdvancedStreamingPipeline::stop() {
    if (!is_running_.load()) {
        return;
    }

    is_running_.store(false);

    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }

    auto& logger = StructuredLogger::instance();
    logger.log_info("Streaming pipeline stopped");
}

bool AdvancedStreamingPipeline::push_stereo_pair(const cv::Mat& left, const cv::Mat& right) {
    if (!is_running_.load()) {
        return false;
    }

    // For simplicity, we'll use the same buffer for both left and right frames
    // In a production system, you might want separate buffers or a paired buffer
    return frame_buffer_->push_frame(left) && frame_buffer_->push_frame(right);
}

bool AdvancedStreamingPipeline::get_latest_result(cv::Mat& disparity_map, std::chrono::milliseconds max_wait) {
    std::unique_lock<std::mutex> lock(results_mutex_);

    if (latest_disparity_.empty()) {
        return false;
    }

    disparity_map = latest_disparity_.clone();
    return true;
}

void AdvancedStreamingPipeline::processing_loop() {
    auto& logger = StructuredLogger::instance();
    logger.log_info("Processing loop started");

    auto target_frame_time = std::chrono::microseconds(
        static_cast<long long>(1000000.0 / config_.target_fps));

    while (is_running_.load()) {
        auto loop_start = std::chrono::high_resolution_clock::now();

        AdvancedFrameBuffer::FrameData left_data, right_data;

        // Get frames from buffer
        if (frame_buffer_->pop_frame(left_data) && frame_buffer_->pop_frame(right_data)) {
            try {
                process_frame_pair(left_data, right_data);
            } catch (const std::exception& e) {
                logger.log_error("Frame processing failed", {
                    {"error", e.what()}
                });
            }
        }

        // Maintain target frame rate
        auto elapsed = std::chrono::high_resolution_clock::now() - loop_start;
        if (elapsed < target_frame_time) {
            std::this_thread::sleep_for(target_frame_time - elapsed);
        }
    }

    logger.log_info("Processing loop stopped");
}

void AdvancedStreamingPipeline::monitoring_loop() {
    auto& logger = StructuredLogger::instance();

    while (is_running_.load()) {
        update_performance_stats();

        if (config_.enable_adaptive_quality) {
            adapt_quality_if_needed();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // Update every second
    }
}

void AdvancedStreamingPipeline::process_frame_pair(const AdvancedFrameBuffer::FrameData& left_data,
                                                  const AdvancedFrameBuffer::FrameData& right_data) {

    auto processing_start = std::chrono::high_resolution_clock::now();

    // Process stereo pair
    cv::Mat disparity = gpu_processor_->process_stereo_sync(left_data.cpu_frame, right_data.cpu_frame);

    if (!disparity.empty()) {
        // Update results
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            latest_disparity_ = disparity.clone();
            last_result_time_ = std::chrono::high_resolution_clock::now();
        }

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_frames_processed++;

            auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - processing_start);

            stats_.average_latency_ms =
                (stats_.average_latency_ms * (stats_.total_frames_processed - 1) +
                 processing_time.count()) / stats_.total_frames_processed;
        }
    }
}

cv::Mat AdvancedStreamingPipeline::generate_point_cloud(const cv::Mat& disparity_map, const cv::Mat& left_frame) {
    // Simplified point cloud generation - in production you'd use proper calibration
    cv::Mat point_cloud;
    // Implementation would go here...
    return point_cloud;
}

void AdvancedStreamingPipeline::update_performance_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Calculate FPS
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

    stats_.current_fps = static_cast<double>(frame_timestamps_.size());
    stats_.buffer_stats = frame_buffer_->get_stats();
    stats_.processing_stats = gpu_processor_->get_stats();
}

void AdvancedStreamingPipeline::adapt_quality_if_needed() {
    // Simple quality adaptation based on processing load
    std::lock_guard<std::mutex> lock(stats_mutex_);

    double load_ratio = stats_.average_latency_ms / config_.max_latency_ms;

    if (load_ratio > config_.quality_adaptation_threshold) {
        // Decrease quality
        if (current_quality_index_ > 0) {
            current_quality_index_--;
            // Would update processing resolution here
        }
    } else if (load_ratio < config_.quality_adaptation_threshold * 0.5) {
        // Increase quality
        if (current_quality_index_ < config_.quality_levels.size() - 1) {
            current_quality_index_++;
            // Would update processing resolution here
        }
    }
}

AdvancedStreamingPipeline::PipelineStats AdvancedStreamingPipeline::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool AdvancedStreamingPipeline::is_running() const {
    return is_running_.load();
}

// StreamingPipelineFactory Implementation
StreamingPipelineFactory::HardwareProfile StreamingPipelineFactory::detect_hardware() {
    HardwareProfile profile;

#ifdef USE_CUDA
    int device_count;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        profile.has_cuda = true;

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            profile.gpu_memory_mb = prop.totalGlobalMem / (1024 * 1024);
            profile.compute_capability_major = prop.major;
            profile.compute_capability_minor = prop.minor;
        }
    }
#endif

#ifdef USE_HIP
    int device_count;
    if (hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0) {
        profile.has_hip = true;
    }
#endif

    profile.cpu_cores = std::thread::hardware_concurrency();

    auto& logger = StructuredLogger::instance();
    logger.log_info("Hardware profile detected", {
        {"has_cuda", profile.has_cuda},
        {"has_hip", profile.has_hip},
        {"gpu_memory_mb", profile.gpu_memory_mb},
        {"cpu_cores", profile.cpu_cores}
    });

    return profile;
}

AdvancedStreamingPipeline::PipelineConfig StreamingPipelineFactory::create_optimized_config(const HardwareProfile& profile) {
    AdvancedStreamingPipeline::PipelineConfig config;

    if (profile.has_cuda && profile.gpu_memory_mb > 2048) {
        // High-end GPU configuration
        config.buffer_config.max_buffer_size = 15;
        config.buffer_config.gpu_buffer_size = 5;
        config.processing_config.num_cuda_streams = 6;
        config.target_fps = 60.0;
    } else if (profile.has_cuda || profile.has_hip) {
        // Mid-range GPU configuration
        config.buffer_config.max_buffer_size = 10;
        config.buffer_config.gpu_buffer_size = 3;
        config.processing_config.num_cuda_streams = 4;
        config.target_fps = 30.0;
    } else {
        // CPU-only configuration
        config.buffer_config.max_buffer_size = 8;
        config.buffer_config.gpu_buffer_size = 0;
        config.buffer_config.enable_gpu_upload = false;
        config.target_fps = 15.0;
    }

    return config;
}

std::unique_ptr<AdvancedStreamingPipeline> StreamingPipelineFactory::create_pipeline(const HardwareProfile& profile) {
    auto config = create_optimized_config(profile);
    return std::make_unique<AdvancedStreamingPipeline>(config);
}

} // namespace streaming
} // namespace cv_stereo
