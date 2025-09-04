#pragma once

#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <future>
#include <vector>
#include <map>

// Forward declarations to avoid compile-time dependency issues
namespace cv {
    class Mat;
    class Size;
    template<typename T> class Ptr;
    class StereoBM;
    namespace cuda {
        class GpuMat;
        class StereoBM;
    }
}

#ifdef USE_CUDA
struct cudaStream_t;
#endif

#ifdef USE_HIP
struct hipStream_t;
#endif

#include "logging/structured_logger.hpp"

namespace cv_stereo {
namespace streaming {

/**
 * @brief Advanced frame buffer with multi-level caching and GPU memory management
 */
class AdvancedFrameBuffer {
public:
    struct FrameData {
        cv::Mat cpu_frame;
        std::chrono::high_resolution_clock::time_point timestamp;
        uint64_t sequence_number;

#ifdef USE_CUDA
        cv::cuda::GpuMat gpu_frame;
        cudaStream_t cuda_stream;
#endif

        bool is_processed = false;
        std::future<cv::Mat> processing_future;
    };

    struct BufferConfig {
        size_t max_buffer_size = 10;
        size_t gpu_buffer_size = 3;  // Smaller GPU buffer due to memory constraints
        bool enable_gpu_upload = true;
        bool enable_async_upload = true;
        size_t drop_threshold = 8;  // Drop frames when buffer reaches this size
        std::chrono::milliseconds max_frame_age{500};  // Drop frames older than this
    };

    explicit AdvancedFrameBuffer(const BufferConfig& config = BufferConfig{});
    ~AdvancedFrameBuffer();

    // Frame management
    bool push_frame(const cv::Mat& frame);
    bool pop_frame(FrameData& frame_data);
    bool peek_latest_frame(FrameData& frame_data);

    // Buffer management
    size_t size() const;
    size_t gpu_size() const;
    bool is_full() const;
    void clear();

    // Statistics
    struct BufferStats {
        size_t total_frames_pushed = 0;
        size_t total_frames_dropped = 0;
        size_t gpu_upload_failures = 0;
        double average_age_ms = 0.0;
        size_t current_size = 0;
        size_t peak_size = 0;
    };

    BufferStats get_stats() const;

private:
    BufferConfig config_;
    mutable std::mutex buffer_mutex_;
    std::condition_variable buffer_condition_;

    std::queue<FrameData> cpu_buffer_;
    std::queue<FrameData> gpu_buffer_;

    std::atomic<uint64_t> sequence_counter_{0};
    BufferStats stats_;

    void cleanup_old_frames();
    bool upload_to_gpu(FrameData& frame_data);

#ifdef USE_CUDA
    std::vector<cudaStream_t> upload_streams_;
    void initialize_cuda_streams();
    void cleanup_cuda_streams();
#endif
};

/**
 * @brief GPU-accelerated stereo processing pipeline with stream overlap
 */
class GPUStreamProcessor {
public:
    struct ProcessingConfig {
        int num_cuda_streams = 4;
        int num_hip_streams = 4;
        bool enable_stream_overlap = true;
        bool enable_async_processing = true;
        bool enable_result_caching = true;

        // Stereo processing parameters
        int max_disparity = 64;
        int block_size = 15;
        int min_disparity = 0;

        // Performance tuning
        cv::Size processing_size{640, 480};  // Resize for performance
        bool enable_preprocessing = true;
        bool enable_postprocessing = true;
    };

    explicit GPUStreamProcessor(const ProcessingConfig& config = ProcessingConfig{});
    ~GPUStreamProcessor();

    // Processing interface
    std::future<cv::Mat> process_stereo_async(const cv::Mat& left, const cv::Mat& right);
    cv::Mat process_stereo_sync(const cv::Mat& left, const cv::Mat& right);

    // Batch processing for efficiency
    std::vector<std::future<cv::Mat>> process_batch_async(
        const std::vector<std::pair<cv::Mat, cv::Mat>>& stereo_pairs);

    // Performance monitoring
    struct ProcessingStats {
        double average_processing_time_ms = 0.0;
        size_t total_processed_frames = 0;
        size_t gpu_memory_used_mb = 0;
        double gpu_utilization_percent = 0.0;
        size_t stream_queue_depth = 0;
    };

    ProcessingStats get_stats() const;

    // Resource management
    bool is_gpu_available() const;
    void flush_all_streams();
    void clear_cache();

private:
    ProcessingConfig config_;
    ProcessingStats stats_;

#ifdef USE_CUDA
    std::vector<cudaStream_t> cuda_streams_;
    cv::Ptr<cv::cuda::StereoBM> cuda_stereo_matcher_;
    std::vector<cv::cuda::GpuMat> gpu_memory_pool_;
#endif

#ifdef USE_HIP
    std::vector<hipStream_t> hip_streams_;
    // HIP-specific processing components
#endif

    // CPU fallback
    cv::Ptr<cv::StereoBM> cpu_stereo_matcher_;

    // Stream management
    std::atomic<size_t> current_stream_index_{0};
    std::mutex processing_mutex_;

    // Result caching
    std::map<std::pair<size_t, size_t>, cv::Mat> result_cache_;
    std::mutex cache_mutex_;

    void initialize_gpu_resources();
    void cleanup_gpu_resources();
    size_t get_next_stream_index();
    cv::Mat process_on_stream(const cv::Mat& left, const cv::Mat& right, size_t stream_index);

    // Hash function for frame pair caching
    std::pair<size_t, size_t> compute_frame_hash(const cv::Mat& left, const cv::Mat& right);
};

/**
 * @brief Advanced streaming pipeline with intelligent buffering and GPU optimization
 */
class AdvancedStreamingPipeline {
public:
    struct PipelineConfig {
        // Buffer configuration
        AdvancedFrameBuffer::BufferConfig buffer_config;

        // Processing configuration
        GPUStreamProcessor::ProcessingConfig processing_config;

        // Pipeline behavior
        bool enable_frame_dropping = true;
        bool enable_adaptive_quality = true;
        bool enable_predictive_processing = true;

        // Performance targets
        double target_fps = 30.0;
        double max_latency_ms = 100.0;

        // Quality adaptation
        std::vector<cv::Size> quality_levels = {{320, 240}, {640, 480}, {1280, 720}};
        double quality_adaptation_threshold = 0.8;  // Adapt when processing load > 80%
    };

    explicit AdvancedStreamingPipeline(const PipelineConfig& config = PipelineConfig{});
    ~AdvancedStreamingPipeline();

    // Pipeline control
    bool start();
    void stop();
    bool is_running() const;

    // Frame input
    bool push_stereo_pair(const cv::Mat& left, const cv::Mat& right);

    // Result retrieval
    bool get_latest_result(cv::Mat& disparity_map, std::chrono::milliseconds max_wait = std::chrono::milliseconds(50));
    bool get_latest_point_cloud(cv::Mat& point_cloud, std::chrono::milliseconds max_wait = std::chrono::milliseconds(50));

    // Performance monitoring and adaptation
    struct PipelineStats {
        double current_fps = 0.0;
        double average_latency_ms = 0.0;
        double gpu_utilization_percent = 0.0;
        size_t frames_dropped = 0;
        size_t total_frames_processed = 0;
        cv::Size current_quality_level{640, 480};

        AdvancedFrameBuffer::BufferStats buffer_stats;
        GPUStreamProcessor::ProcessingStats processing_stats;
    };

    PipelineStats get_stats() const;

    // Configuration updates
    void update_target_fps(double fps);
    void update_quality_level(const cv::Size& size);
    void enable_adaptive_mode(bool enable);

private:
    PipelineConfig config_;

    // Core components
    std::unique_ptr<AdvancedFrameBuffer> frame_buffer_;
    std::unique_ptr<GPUStreamProcessor> gpu_processor_;

    // Pipeline threads
    std::thread processing_thread_;
    std::thread monitoring_thread_;
    std::atomic<bool> is_running_{false};

    // Results
    std::mutex results_mutex_;
    cv::Mat latest_disparity_;
    cv::Mat latest_point_cloud_;
    std::chrono::high_resolution_clock::time_point last_result_time_;

    // Performance monitoring
    mutable std::mutex stats_mutex_;
    PipelineStats stats_;
    std::deque<std::chrono::high_resolution_clock::time_point> frame_timestamps_;

    // Adaptive quality control
    std::atomic<size_t> current_quality_index_{1};  // Start with medium quality
    std::chrono::high_resolution_clock::time_point last_adaptation_time_;

    // Private methods
    void processing_loop();
    void monitoring_loop();
    void update_performance_stats();
    void adapt_quality_if_needed();
    void process_frame_pair(const AdvancedFrameBuffer::FrameData& left_data,
                          const AdvancedFrameBuffer::FrameData& right_data);

    cv::Mat generate_point_cloud(const cv::Mat& disparity_map, const cv::Mat& left_frame);
};

/**
 * @brief Factory for creating optimized streaming pipelines based on hardware capabilities
 */
class StreamingPipelineFactory {
public:
    struct HardwareProfile {
        bool has_cuda = false;
        bool has_hip = false;
        size_t gpu_memory_mb = 0;
        int compute_capability_major = 0;
        int compute_capability_minor = 0;
        size_t cpu_cores = 0;
        size_t system_memory_mb = 0;
    };

    static HardwareProfile detect_hardware();
    static AdvancedStreamingPipeline::PipelineConfig create_optimized_config(const HardwareProfile& profile);
    static std::unique_ptr<AdvancedStreamingPipeline> create_pipeline(const HardwareProfile& profile);

    // Preset configurations
    static AdvancedStreamingPipeline::PipelineConfig get_realtime_config();
    static AdvancedStreamingPipeline::PipelineConfig get_quality_config();
    static AdvancedStreamingPipeline::PipelineConfig get_balanced_config();
};

} // namespace cv_stereo::streaming
