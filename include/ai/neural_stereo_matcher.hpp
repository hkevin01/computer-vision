#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>

#ifdef WITH_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#endif

#ifdef WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace stereo_vision::ai {

/**
 * @brief Neural network-based stereo matching using TensorRT/ONNX Runtime
 * 
 * This class provides high-performance neural network inference for stereo depth estimation.
 * It supports multiple backends (TensorRT for NVIDIA GPUs, ONNX Runtime for CPU/GPU).
 */
class NeuralStereoMatcher {
public:
    enum class Backend {
        AUTO,           ///< Automatically select best available backend
        TENSORRT,       ///< NVIDIA TensorRT (GPU only)
        ONNX_GPU,       ///< ONNX Runtime with GPU provider
        ONNX_CPU        ///< ONNX Runtime with CPU provider
    };

    enum class ModelType {
        HITNET,         ///< HITNet model for real-time stereo
        RAFT_STEREO,    ///< RAFT-Stereo for high accuracy
        STTR,           ///< Stereo Transformer
        CUSTOM          ///< User-provided model
    };

    struct ModelConfig {
        ModelType type = ModelType::HITNET;
        std::string model_path;
        int input_width = 640;
        int input_height = 480;
        float max_disparity = 192.0f;
        bool use_fp16 = true;           ///< Use half precision for faster inference
        int batch_size = 1;
        Backend preferred_backend = Backend::AUTO;
    };

    struct InferenceStats {
        double preprocessing_time_ms = 0.0;
        double inference_time_ms = 0.0;
        double postprocessing_time_ms = 0.0;
        double total_time_ms = 0.0;
        double fps = 0.0;
        size_t memory_usage_mb = 0;
        std::string backend_used;
    };

public:
    explicit NeuralStereoMatcher(const ModelConfig& config = ModelConfig{});
    ~NeuralStereoMatcher();

    // Non-copyable but movable
    NeuralStereoMatcher(const NeuralStereoMatcher&) = delete;
    NeuralStereoMatcher& operator=(const NeuralStereoMatcher&) = delete;
    NeuralStereoMatcher(NeuralStereoMatcher&&) = default;
    NeuralStereoMatcher& operator=(NeuralStereoMatcher&&) = default;

    /**
     * @brief Initialize the neural network model
     * @param config Model configuration
     * @return true if initialization successful
     */
    bool initialize(const ModelConfig& config);

    /**
     * @brief Check if the matcher is ready for inference
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Compute disparity map using neural network
     * @param left_image Left stereo image
     * @param right_image Right stereo image
     * @return Disparity map (CV_32F)
     */
    cv::Mat computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image);

    /**
     * @brief Compute disparity map with confidence
     * @param left_image Left stereo image
     * @param right_image Right stereo image
     * @param confidence Output confidence map (CV_32F, 0-1 range)
     * @return Disparity map (CV_32F)
     */
    cv::Mat computeDisparityWithConfidence(const cv::Mat& left_image, 
                                         const cv::Mat& right_image,
                                         cv::Mat& confidence);

    /**
     * @brief Get performance statistics from last inference
     */
    const InferenceStats& getLastStats() const { return last_stats_; }

    /**
     * @brief Get current model configuration
     */
    const ModelConfig& getConfig() const { return config_; }

    /**
     * @brief Get available backends on this system
     */
    static std::vector<Backend> getAvailableBackends();

    /**
     * @brief Download and setup pre-trained models
     * @param model_type Type of model to download
     * @param output_dir Directory to save model files
     * @return Path to downloaded model file
     */
    static std::string downloadPretrainedModel(ModelType model_type, 
                                             const std::string& output_dir = "models/");

    /**
     * @brief Benchmark different models and backends
     * @param test_images Pairs of test images for benchmarking
     * @return Performance comparison results
     */
    static std::map<std::string, InferenceStats> benchmarkModels(
        const std::vector<std::pair<cv::Mat, cv::Mat>>& test_images);

private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl_;

    ModelConfig config_;
    bool initialized_ = false;
    mutable InferenceStats last_stats_;

    // Helper methods
    cv::Mat preprocessImages(const cv::Mat& left, const cv::Mat& right);
    cv::Mat postprocessDisparity(const std::vector<float>& raw_output);
    Backend selectOptimalBackend();
    void updateStats(const std::chrono::high_resolution_clock::time_point& start_time);
};

/**
 * @brief Factory class for creating neural stereo matchers
 */
class NeuralMatcherFactory {
public:
    /**
     * @brief Create matcher with automatic model selection
     * @param performance_target Target performance level (fps)
     * @param quality_target Target quality level (0-1)
     * @return Configured neural stereo matcher
     */
    static std::unique_ptr<NeuralStereoMatcher> createOptimalMatcher(
        double performance_target_fps = 30.0,
        double quality_target = 0.8);

    /**
     * @brief Create matcher for real-time applications
     */
    static std::unique_ptr<NeuralStereoMatcher> createRealtimeMatcher();

    /**
     * @brief Create matcher for high-quality applications
     */
    static std::unique_ptr<NeuralStereoMatcher> createHighQualityMatcher();

    /**
     * @brief List all available pre-trained models
     */
    static std::vector<std::pair<ModelType, std::string>> getAvailableModels();
};

/**
 * @brief Adaptive neural stereo matcher that automatically adjusts quality based on performance
 */
class AdaptiveNeuralMatcher {
public:
    struct AdaptiveConfig {
        double target_fps = 30.0;
        double min_quality = 0.5;
        double max_quality = 0.95;
        bool enable_dynamic_resolution = true;
        bool enable_model_switching = true;
    };

    explicit AdaptiveNeuralMatcher(const AdaptiveConfig& config = AdaptiveConfig{});

    /**
     * @brief Process stereo pair with automatic quality adjustment
     */
    cv::Mat processAdaptive(const cv::Mat& left, const cv::Mat& right);

    /**
     * @brief Get current adaptation state
     */
    struct AdaptiveState {
        double current_fps;
        double current_quality;
        ModelType current_model;
        cv::Size current_resolution;
        bool is_adapting;
    };

    AdaptiveState getAdaptiveState() const;

private:
    AdaptiveConfig config_;
    std::vector<std::unique_ptr<NeuralStereoMatcher>> matchers_;
    size_t current_matcher_index_ = 0;
    
    // Performance tracking
    std::vector<double> recent_fps_;
    std::vector<double> recent_quality_;
    
    void updatePerformanceHistory(double fps, double quality);
    void adaptConfiguration();
};

} // namespace stereo_vision::ai
