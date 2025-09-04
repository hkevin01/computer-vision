#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cv_stereo {

enum class TensorRTPrecision {
    FP32,
    FP16,
    INT8
};

struct TensorRTCacheKey {
    std::string model_name;
    std::string gpu_arch;    // e.g., "sm_75", "sm_86"
    int onnx_opset_version;
    TensorRTPrecision precision;
    std::vector<int> input_shape;

    std::string to_string() const;
    bool operator==(const TensorRTCacheKey& other) const;
};

struct TensorRTCacheKeyHash {
    std::size_t operator()(const TensorRTCacheKey& key) const;
};

class TensorRTEngineCache {
public:
    static TensorRTEngineCache& instance();

    // Set cache directory (default: data/models/cache)
    void set_cache_directory(const std::string& cache_dir);

    // Check if cached engine exists for given parameters
    bool has_cached_engine(const TensorRTCacheKey& key) const;

    // Get path to cached engine file
    std::string get_cache_path(const TensorRTCacheKey& key) const;

    // Store engine in cache
    bool cache_engine(const TensorRTCacheKey& key, const std::string& engine_data);

    // Load engine from cache
    std::vector<char> load_cached_engine(const TensorRTCacheKey& key) const;

    // Clear cache (optionally filter by model name)
    void clear_cache(const std::string& model_name = "");

    // Get current GPU architecture string
    static std::string get_gpu_architecture();

    // Get cache statistics
    struct CacheStats {
        size_t total_engines = 0;
        size_t total_size_bytes = 0;
        std::unordered_map<std::string, size_t> engines_per_model;
    };
    CacheStats get_cache_stats() const;

private:
    TensorRTEngineCache() = default;

    std::string cache_directory_ = "data/models/cache";
    mutable std::unordered_map<TensorRTCacheKey, std::string, TensorRTCacheKeyHash> cache_paths_;

    // Helper methods
    std::string generate_cache_filename(const TensorRTCacheKey& key) const;
    bool ensure_cache_directory() const;
};

// Helper class for TensorRT session management with fallback
class TensorRTSessionManager {
public:
    struct SessionConfig {
        std::string model_name;
        std::string onnx_path;
        TensorRTPrecision precision = TensorRTPrecision::FP16;
        std::vector<int> input_shape;
        bool enable_cache = true;
        bool fallback_to_onnx = true;
        std::vector<std::string> fallback_providers = {"CUDAExecutionProvider", "CPUExecutionProvider"};
    };

    // Create or load TensorRT engine with caching
    bool create_session(const SessionConfig& config, std::string& error_msg);

    // Run inference (implementation depends on your inference framework)
    bool run_inference(const std::vector<float>& input_data, std::vector<float>& output_data, std::string& error_msg);

    // Get current provider being used
    std::string get_current_provider() const { return current_provider_; }

    // Check if TensorRT is available
    static bool is_tensorrt_available();

private:
    std::string current_provider_;
    SessionConfig config_;
    bool session_ready_ = false;

    // Implementation-specific session handle (void* for now)
    void* session_handle_ = nullptr;
};

} // namespace cv_stereo
