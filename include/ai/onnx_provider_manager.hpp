#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#ifdef CV_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

// Forward declarations to avoid including heavy headers
namespace cv {
    class Mat;
}

namespace cv_stereo {

enum class ONNXProvider {
    CPU,
    CUDA,
    DirectML,     // Windows
    CoreML,       // macOS
    TensorRT
};

struct ONNXSessionConfig {
    ONNXProvider preferred_provider = ONNXProvider::CPU;
    std::vector<ONNXProvider> fallback_providers;

    // Performance settings
    int num_threads = 0;  // 0 = auto-detect
    bool enable_cpu_mem_arena = true;
    bool enable_memory_pattern = true;

    // GPU settings (CUDA/TensorRT)
    int gpu_device_id = 0;
    size_t gpu_mem_limit = 0;  // 0 = unlimited

    // TensorRT specific
    bool trt_enable_fp16 = false;
    bool trt_enable_int8 = false;
    size_t trt_max_workspace_size = 1ULL << 30;  // 1GB default

    // Model optimization
    bool enable_graph_optimization = true;
    std::string optimization_level;  // "basic", "extended", "all"

    // Constructor with defaults
    ONNXSessionConfig() : optimization_level("basic") {
        fallback_providers.push_back(ONNXProvider::CPU);
    }
};

struct ONNXModelInfo {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int> input_types;  // ONNX tensor types
    std::vector<int> output_types;
};

class ONNXProviderManager {
public:
    static ONNXProviderManager& instance();

    // Provider availability checks
    bool is_provider_available(ONNXProvider provider) const;
    std::vector<ONNXProvider> get_available_providers() const;
    std::string provider_to_string(ONNXProvider provider) const;
    ONNXProvider string_to_provider(const std::string& provider_str) const;

    // Session creation and management
    bool create_session(const std::string& model_path,
                       const ONNXSessionConfig& config,
                       std::string& error_msg);

    bool run_inference(const std::vector<float>& input_data,
                      const std::vector<int64_t>& input_shape,
                      std::vector<float>& output_data,
                      std::vector<int64_t>& output_shape,
                      std::string& error_msg);

    // Model introspection
    bool get_model_info(const std::string& model_path,
                       ONNXModelInfo& info,
                       std::string& error_msg) const;

    // Performance profiling
    struct InferenceStats {
        double preprocessing_time_ms = 0.0;
        double inference_time_ms = 0.0;
        double postprocessing_time_ms = 0.0;
        double total_time_ms = 0.0;
        size_t memory_usage_bytes = 0;
        std::string provider_used;
    };

    const InferenceStats& get_last_inference_stats() const { return last_stats_; }

    // Session management
    void clear_session();
    bool has_active_session() const;
    ONNXProvider get_active_provider() const { return active_provider_; }

private:
    ONNXProviderManager() = default;
    ~ONNXProviderManager() = default;

    bool create_session_with_provider(const std::string& model_path,
                                     ONNXProvider provider,
                                     const ONNXSessionConfig& config,
                                     std::string& error_msg);

    std::vector<ONNXProvider> detect_available_providers() const;

#ifdef CV_WITH_ONNX
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    Ort::SessionOptions session_options_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    ONNXModelInfo model_info_;
#endif

    ONNXProvider active_provider_ = ONNXProvider::CPU;
    ONNXSessionConfig current_config_;
    InferenceStats last_stats_;
    mutable std::vector<ONNXProvider> cached_available_providers_;
    mutable bool providers_cached_ = false;
};

// Utility functions for common operations
namespace onnx_utils {
    // Convert OpenCV Mat to ONNX input format
    std::vector<float> mat_to_onnx_input(const cv::Mat& image,
                                        const std::vector<int64_t>& target_shape,
                                        bool normalize = true,
                                        const std::vector<float>& mean = std::vector<float>{0.485f, 0.456f, 0.406f},
                                        const std::vector<float>& std = std::vector<float>{0.229f, 0.224f, 0.225f});

    // Convert ONNX output to OpenCV Mat
    cv::Mat onnx_output_to_mat(const std::vector<float>& output_data,
                              const std::vector<int64_t>& output_shape,
                              int cv_type = CV_32FC1);

    // Auto-configure session based on available hardware
    ONNXSessionConfig auto_configure_session();

    // Benchmark different providers
    struct ProviderBenchmark {
        ONNXProvider provider;
        double avg_inference_time_ms;
        double memory_usage_mb;
        bool successful;
        std::string error_message;
    };

    std::vector<ProviderBenchmark> benchmark_providers(
        const std::string& model_path,
        const std::vector<int64_t>& input_shape,
        int num_iterations = 10);
}

} // namespace cv_stereo
