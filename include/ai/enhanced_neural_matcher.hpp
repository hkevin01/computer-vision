#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>

#ifdef WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#ifdef WITH_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#endif

#ifdef WITH_OPENVINO
#include <openvino/openvino.hpp>
#endif

namespace stereovision {
namespace ai {

/**
 * @brief Enhanced neural network-based stereo matching with real AI models
 * 
 * This class replaces the placeholder implementation with actual neural network inference
 * supporting multiple backends (ONNX Runtime, TensorRT, OpenVINO) and pre-trained models.
 */
class EnhancedNeuralMatcher {
public:
    enum class Backend {
        AUTO,           ///< Automatically select best available backend
        ONNX_CPU,       ///< ONNX Runtime with CPU provider
        ONNX_GPU,       ///< ONNX Runtime with GPU provider
        TENSORRT,       ///< NVIDIA TensorRT (GPU only)
        OPENVINO        ///< Intel OpenVINO (CPU/GPU/VPU)
    };

    enum class ModelType {
        HITNET,         ///< HITNet - Real-time stereo matching
        RAFT_STEREO,    ///< RAFT-Stereo - High accuracy
        CRESTEREO,      ///< CREStereo - State-of-the-art
        IGEV_STEREO,    ///< IGEV - Iterative geometry
        COEX_NET,       ///< CoEx - Cost-effective
        AANET,          ///< AANet - Adaptive aggregation
        CUSTOM          ///< User-provided model
    };

    struct ModelConfig {
        ModelType type = ModelType::HITNET;
        std::string model_path;
        cv::Size input_size = cv::Size(1280, 720);
        float max_disparity = 192.0f;
        bool use_fp16 = true;           ///< Use half precision for faster inference
        int batch_size = 1;
        Backend preferred_backend = Backend::AUTO;
        
        // Model-specific parameters
        float confidence_threshold = 0.5f;
        bool enable_post_processing = true;
        bool enable_left_right_check = true;
    };

    struct InferenceStats {
        double preprocessing_time_ms = 0.0;
        double inference_time_ms = 0.0;
        double postprocessing_time_ms = 0.0;
        double total_time_ms = 0.0;
        double fps = 0.0;
        size_t memory_usage_mb = 0;
        std::string backend_used;
        std::string model_name;
    };

    struct QualityMetrics {
        float density = 0.0f;           ///< Percentage of valid pixels
        float smoothness = 0.0f;        ///< Local smoothness measure
        float edge_preservation = 0.0f; ///< How well edges are preserved
        float confidence_mean = 0.0f;   ///< Average confidence score
        float confidence_std = 0.0f;    ///< Confidence standard deviation
    };

public:
    EnhancedNeuralMatcher();
    explicit EnhancedNeuralMatcher(const ModelConfig& config);
    ~EnhancedNeuralMatcher();

    // Non-copyable but movable
    EnhancedNeuralMatcher(const EnhancedNeuralMatcher&) = delete;
    EnhancedNeuralMatcher& operator=(const EnhancedNeuralMatcher&) = delete;
    EnhancedNeuralMatcher(EnhancedNeuralMatcher&&) = default;
    EnhancedNeuralMatcher& operator=(EnhancedNeuralMatcher&&) = default;

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
     * @brief Batch processing for efficiency
     * @param left_images Vector of left images
     * @param right_images Vector of right images
     * @return Vector of disparity maps
     */
    std::vector<cv::Mat> computeDisparityBatch(
        const std::vector<cv::Mat>& left_images,
        const std::vector<cv::Mat>& right_images);

    /**
     * @brief Get performance statistics from last inference
     */
    const InferenceStats& getLastStats() const { return last_stats_; }

    /**
     * @brief Get quality metrics from last inference
     */
    const QualityMetrics& getLastQualityMetrics() const { return last_quality_; }

    /**
     * @brief Get current model configuration
     */
    const ModelConfig& getConfig() const { return config_; }

    /**
     * @brief Get available backends on this system
     */
    static std::vector<Backend> getAvailableBackends();

    /**
     * @brief Get model information
     */
    static std::map<ModelType, std::string> getModelInfo();

    /**
     * @brief Auto-select optimal backend based on hardware
     */
    static Backend selectOptimalBackend();

    /**
     * @brief Validate model file integrity
     */
    static bool validateModel(const std::string& model_path);

private:
    // Implementation details hidden in PIMPL
    class Impl;
    std::unique_ptr<Impl> pImpl_;

    ModelConfig config_;
    bool initialized_ = false;
    mutable InferenceStats last_stats_;
    mutable QualityMetrics last_quality_;

    // Helper methods
    cv::Mat preprocessImages(const cv::Mat& left, const cv::Mat& right);
    cv::Mat postprocessDisparity(const cv::Mat& raw_disparity);
    void updateStats(const std::chrono::high_resolution_clock::time_point& start_time,
                    const std::string& backend_name);
    QualityMetrics computeQualityMetrics(const cv::Mat& disparity, 
                                       const cv::Mat& confidence,
                                       const cv::Mat& left_image);
};

/**
 * @brief Factory class for creating enhanced neural stereo matchers
 */
class EnhancedMatcherFactory {
public:
    /**
     * @brief Create matcher with automatic model selection based on performance target
     */
    static std::unique_ptr<EnhancedNeuralMatcher> createOptimalMatcher(
        double target_fps = 30.0,
        double min_accuracy = 0.8);

    /**
     * @brief Create matcher optimized for real-time applications
     */
    static std::unique_ptr<EnhancedNeuralMatcher> createRealtimeMatcher();

    /**
     * @brief Create matcher optimized for high-quality results
     */
    static std::unique_ptr<EnhancedNeuralMatcher> createHighQualityMatcher();

    /**
     * @brief Create matcher for specific hardware
     */
    static std::unique_ptr<EnhancedNeuralMatcher> createForHardware(
        const std::string& gpu_name = "",
        size_t available_memory_mb = 0);

    /**
     * @brief List all available pre-trained models
     */
    static std::vector<std::pair<EnhancedNeuralMatcher::ModelType, std::string>> getAvailableModels();

    /**
     * @brief Download pre-trained model if not present
     */
    static bool downloadModel(EnhancedNeuralMatcher::ModelType type, const std::string& models_dir = "models/");
};

/**
 * @brief Model manager for downloading and validating neural network models
 */
class ModelManager {
public:
    struct ModelInfo {
        std::string name;
        std::string url;
        std::string filename;
        cv::Size input_size;
        float max_disparity;
        float fps_estimate;
        float accuracy_score;
        size_t file_size_mb;
        std::string sha256_hash;
    };

    explicit ModelManager(const std::string& models_dir = "models/");

    /**
     * @brief Download model from remote repository
     */
    bool downloadModel(EnhancedNeuralMatcher::ModelType type, bool force_redownload = false);

    /**
     * @brief Verify model integrity
     */
    bool verifyModel(EnhancedNeuralMatcher::ModelType type);

    /**
     * @brief Get path to downloaded model
     */
    std::string getModelPath(EnhancedNeuralMatcher::ModelType type);

    /**
     * @brief List all available models
     */
    std::vector<ModelInfo> getAvailableModels();

    /**
     * @brief Get model registry
     */
    static const std::map<EnhancedNeuralMatcher::ModelType, ModelInfo>& getModelRegistry();

private:
    std::string models_dir_;
    static const std::map<EnhancedNeuralMatcher::ModelType, ModelInfo> model_registry_;
    
    bool downloadFromUrl(const std::string& url, const std::string& output_path);
    bool verifyChecksum(const std::string& file_path, const std::string& expected_hash);
};

} // namespace ai
} // namespace stereovision
