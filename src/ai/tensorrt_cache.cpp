#include "ai/tensorrt_cache.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#ifdef CV_WITH_TENSORRT
#include <cuda_runtime.h>
#endif

namespace cv_stereo {

std::string TensorRTCacheKey::to_string() const {
    std::stringstream ss;
    ss << model_name << "_" << gpu_arch << "_opset" << onnx_opset_version << "_";

    switch (precision) {
        case TensorRTPrecision::FP32: ss << "fp32"; break;
        case TensorRTPrecision::FP16: ss << "fp16"; break;
        case TensorRTPrecision::INT8: ss << "int8"; break;
    }

    ss << "_shape";
    for (int dim : input_shape) {
        ss << "_" << dim;
    }

    return ss.str();
}

bool TensorRTCacheKey::operator==(const TensorRTCacheKey& other) const {
    return model_name == other.model_name &&
           gpu_arch == other.gpu_arch &&
           onnx_opset_version == other.onnx_opset_version &&
           precision == other.precision &&
           input_shape == other.input_shape;
}

std::size_t TensorRTCacheKeyHash::operator()(const TensorRTCacheKey& key) const {
    std::size_t h1 = std::hash<std::string>{}(key.model_name);
    std::size_t h2 = std::hash<std::string>{}(key.gpu_arch);
    std::size_t h3 = std::hash<int>{}(key.onnx_opset_version);
    std::size_t h4 = std::hash<int>{}(static_cast<int>(key.precision));

    std::size_t h5 = 0;
    for (int dim : key.input_shape) {
        h5 ^= std::hash<int>{}(dim) + 0x9e3779b9 + (h5 << 6) + (h5 >> 2);
    }

    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
}

TensorRTEngineCache& TensorRTEngineCache::instance() {
    static TensorRTEngineCache instance;
    return instance;
}

void TensorRTEngineCache::set_cache_directory(const std::string& cache_dir) {
    cache_directory_ = cache_dir;
    cache_paths_.clear(); // Invalidate cached paths
}

bool TensorRTEngineCache::has_cached_engine(const TensorRTCacheKey& key) const {
    std::string cache_path = get_cache_path(key);
    std::ifstream file(cache_path, std::ios::binary);
    return file.good();
}

std::string TensorRTEngineCache::get_cache_path(const TensorRTCacheKey& key) const {
    auto it = cache_paths_.find(key);
    if (it != cache_paths_.end()) {
        return it->second;
    }

    std::string filename = generate_cache_filename(key);
    std::string full_path = cache_directory_ + "/" + filename;
    cache_paths_[key] = full_path;

    return full_path;
}

bool TensorRTEngineCache::cache_engine(const TensorRTCacheKey& key, const std::string& engine_data) {
    if (!ensure_cache_directory()) {
        return false;
    }

    std::string cache_path = get_cache_path(key);
    std::ofstream file(cache_path, std::ios::binary);
    if (!file) {
        return false;
    }

    file.write(engine_data.data(), engine_data.size());
    return file.good();
}

std::vector<char> TensorRTEngineCache::load_cached_engine(const TensorRTCacheKey& key) const {
    std::string cache_path = get_cache_path(key);
    std::ifstream file(cache_path, std::ios::binary | std::ios::ate);

    if (!file) {
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        return {};
    }

    return buffer;
}

void TensorRTEngineCache::clear_cache(const std::string& model_name) {
    // Implementation would require filesystem operations
    // For now, just clear the in-memory cache paths
    if (model_name.empty()) {
        cache_paths_.clear();
    } else {
        auto it = cache_paths_.begin();
        while (it != cache_paths_.end()) {
            if (it->first.model_name == model_name) {
                it = cache_paths_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

std::string TensorRTEngineCache::get_gpu_architecture() {
#ifdef CV_WITH_TENSORRT
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    return "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
#else
    return "unknown";
#endif
}

TensorRTEngineCache::CacheStats TensorRTEngineCache::get_cache_stats() const {
    CacheStats stats;

    for (const auto& [key, path] : cache_paths_) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (file) {
            stats.total_engines++;
            std::streamsize size = file.tellg();
            stats.total_size_bytes += size;
            stats.engines_per_model[key.model_name]++;
        }
    }

    return stats;
}

std::string TensorRTEngineCache::generate_cache_filename(const TensorRTCacheKey& key) const {
    return key.to_string() + ".trt";
}

bool TensorRTEngineCache::ensure_cache_directory() const {
    // Would need filesystem operations here
    // For now, assume directory exists
    return true;
}

// TensorRTSessionManager implementation
bool TensorRTSessionManager::create_session(const SessionConfig& config, std::string& error_msg) {
    config_ = config;

    // Check if TensorRT is available
    if (!is_tensorrt_available()) {
        if (config.fallback_to_onnx) {
            current_provider_ = "CUDAExecutionProvider";  // or first fallback
            error_msg = "TensorRT not available, falling back to " + current_provider_;
            return true;  // Continue with fallback
        } else {
            error_msg = "TensorRT not available and fallback disabled";
            return false;
        }
    }

    // Try to load cached engine
    TensorRTCacheKey cache_key;
    cache_key.model_name = config.model_name;
    cache_key.gpu_arch = TensorRTEngineCache::get_gpu_architecture();
    cache_key.onnx_opset_version = 11;  // Default, should be detected from ONNX
    cache_key.precision = config.precision;
    cache_key.input_shape = config.input_shape;

    auto& cache = TensorRTEngineCache::instance();

    if (config.enable_cache && cache.has_cached_engine(cache_key)) {
        auto engine_data = cache.load_cached_engine(cache_key);
        if (!engine_data.empty()) {
            // Load cached engine (implementation specific)
            current_provider_ = "TensorRTExecutionProvider";
            session_ready_ = true;
            return true;
        }
    }

    // Build new engine and cache it
    // This would involve actual TensorRT engine building
    // For now, simulate success
    current_provider_ = "TensorRTExecutionProvider";
    session_ready_ = true;

    return true;
}

bool TensorRTSessionManager::run_inference(const std::vector<float>& input_data,
                                          std::vector<float>& output_data,
                                          std::string& error_msg) {
    if (!session_ready_) {
        error_msg = "Session not ready";
        return false;
    }

    // Implementation would depend on your inference framework
    // This is a placeholder
    output_data.resize(input_data.size());  // Dummy implementation
    std::copy(input_data.begin(), input_data.end(), output_data.begin());

    return true;
}

bool TensorRTSessionManager::is_tensorrt_available() {
#ifdef CV_WITH_TENSORRT
    // Check if TensorRT libraries are available and CUDA device is present
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

} // namespace cv_stereo
