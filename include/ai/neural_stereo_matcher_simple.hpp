#pragma once

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace stereovision {
namespace ai {

enum class Backend {
    AUTO = 0,
    TENSORRT = 1,
    ONNX_GPU = 2,
    ONNX_CPU = 3
};

enum class ModelType {
    STEREONET = 0,
    PSM_NET = 1,
    GA_NET = 2,
    HITNET = 3,
    CUSTOM = 4
};

struct ModelConfig {
    ModelType model_type;
    Backend preferred_backend;
    std::string model_path;
    int input_width;
    int input_height;
    int max_disparity;
    bool enable_confidence;
    
    ModelConfig() : model_type(ModelType::STEREONET), preferred_backend(Backend::AUTO),
                   model_path(""), input_width(640), input_height(480), 
                   max_disparity(128), enable_confidence(false) {}
};

struct InferenceStats {
    double avg_fps;
    double peak_fps;
    int total_frames;
    double memory_usage_mb;
    
    InferenceStats() : avg_fps(0.0), peak_fps(0.0), total_frames(0), memory_usage_mb(0.0) {}
};

class NeuralStereoMatcher {
public:
    explicit NeuralStereoMatcher(const ModelConfig& config = ModelConfig{});
    virtual ~NeuralStereoMatcher() = default;

    // Core functionality
    bool initialize(const ModelConfig& config);
    cv::Mat computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image);
    cv::Mat computeDisparityWithConfidence(const cv::Mat& left_image, const cv::Mat& right_image, cv::Mat& confidence_map);
    
    // Backend management
    static std::vector<Backend> getAvailableBackends();
    
    // Model management
    static std::string downloadPretrainedModel(ModelType model_type, const std::string& download_path = "models/");
    static std::vector<std::pair<ModelType, std::string>> getAvailableModels();
    
    // Performance testing
    std::vector<InferenceStats> benchmarkModels(const std::vector<ModelType>& models, const cv::Mat& test_left, const cv::Mat& test_right);
    
    // Statistics
    InferenceStats getStats() const { return current_stats_; }
    
private:
    ModelConfig config_;
    Backend active_backend_;
    bool initialized_;
    InferenceStats current_stats_;
    
    // Implementation methods
    Backend selectOptimalBackend();
    cv::Mat preprocessImages(const cv::Mat& left, const cv::Mat& right);
    cv::Mat postprocessDisparity(const std::vector<float>& raw_output);
    void updateStats(const std::chrono::high_resolution_clock::time_point& start_time);
};

// Factory for creating optimal configurations
class NeuralMatcherFactory {
public:
    static std::unique_ptr<NeuralStereoMatcher> createOptimalMatcher(const cv::Size& input_size = cv::Size(640, 480));
    static std::unique_ptr<NeuralStereoMatcher> createRealtimeMatcher();
    static std::unique_ptr<NeuralStereoMatcher> createHighQualityMatcher();
    static std::vector<std::pair<ModelType, std::string>> getAvailableModels();
};

// Adaptive matcher that switches models based on performance
struct AdaptiveConfig {
    double target_fps;
    double quality_threshold;
    int history_size;
    
    AdaptiveConfig() : target_fps(30.0), quality_threshold(0.8), history_size(10) {}
};

class AdaptiveNeuralMatcher {
public:
    explicit AdaptiveNeuralMatcher(const AdaptiveConfig& config = AdaptiveConfig{});
    cv::Mat processAdaptive(const cv::Mat& left, const cv::Mat& right);
    
    struct AdaptiveState {
        int active_matcher_index;
        double current_fps;
        double current_quality;
        std::vector<double> performance_history;
        
        AdaptiveState() : active_matcher_index(0), current_fps(0.0), current_quality(0.0) {}
    };
    
    AdaptiveState getAdaptiveState() const { return adaptive_state_; }
    
private:
    AdaptiveConfig config_;
    std::vector<std::unique_ptr<NeuralStereoMatcher>> matchers_;
    int current_matcher_index_;
    std::vector<double> recent_fps_;
    std::vector<double> recent_quality_;
    AdaptiveState adaptive_state_;
    
    void updatePerformanceHistory(double fps, double quality);
    void adaptConfiguration();
};

} // namespace ai
} // namespace stereovision
