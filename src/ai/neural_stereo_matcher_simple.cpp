#include "ai/neural_stereo_matcher_simple.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace stereovision {
namespace ai {

NeuralStereoMatcher::NeuralStereoMatcher(const ModelConfig& config) 
    : config_(config), active_backend_(Backend::AUTO), initialized_(false) {
}

bool NeuralStereoMatcher::initialize(const ModelConfig& config) {
    config_ = config;
    
    // Select backend
    if (config_.preferred_backend == Backend::AUTO) {
        active_backend_ = selectOptimalBackend();
    } else {
        active_backend_ = config_.preferred_backend;
    }
    
    // Initialize based on backend
    switch (active_backend_) {
        case Backend::TENSORRT:
            std::cout << "Initializing TensorRT backend (simulation)\n";
            break;
        case Backend::ONNX_GPU:
        case Backend::ONNX_CPU:
            std::cout << "Initializing ONNX backend (simulation)\n";
            break;
        default:
            std::cout << "Using OpenCV stereo matching as fallback\n";
    }
    
    initialized_ = true;
    return true;
}

cv::Mat NeuralStereoMatcher::computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image) {
    if (!initialized_) {
        std::cerr << "NeuralStereoMatcher not initialized\n";
        return cv::Mat();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cv::Mat disparity;
    
    // For now, use OpenCV's stereo matching as simulation
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(config_.max_disparity, 21);
    
    cv::Mat left_gray, right_gray;
    if (left_image.channels() == 3) {
        cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    } else {
        left_gray = left_image;
        right_gray = right_image;
    }
    
    stereo->compute(left_gray, right_gray, disparity);
    disparity.convertTo(disparity, CV_32F, 1.0/16.0);
    
    updateStats(start_time);
    return disparity;
}

cv::Mat NeuralStereoMatcher::computeDisparityWithConfidence(const cv::Mat& left_image, 
                                                           const cv::Mat& right_image, 
                                                           cv::Mat& confidence_map) {
    cv::Mat disparity = computeDisparity(left_image, right_image);
    
    // Simple confidence map generation
    confidence_map = cv::Mat::ones(disparity.size(), CV_32F);
    cv::Mat mask = (disparity > 0);
    confidence_map.setTo(0.8, mask);
    
    return disparity;
}

std::vector<Backend> NeuralStereoMatcher::getAvailableBackends() {
    std::vector<Backend> backends;
    backends.push_back(Backend::ONNX_CPU);  // Always available with OpenCV
    
    // Check for GPU support
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        backends.push_back(Backend::ONNX_GPU);
    }
    
    return backends;
}

std::string NeuralStereoMatcher::downloadPretrainedModel(ModelType model_type, const std::string& download_path) {
    std::string model_name;
    switch (model_type) {
        case ModelType::STEREONET:
            model_name = "stereonet_kitti.onnx";
            break;
        case ModelType::PSM_NET:
            model_name = "psmnet_sceneflow.onnx";
            break;
        default:
            model_name = "default_stereo.onnx";
    }
    
    std::string full_path = download_path + model_name;
    std::cout << "Simulated download of " << model_name << " to " << full_path << std::endl;
    
    return full_path;
}

std::vector<std::pair<ModelType, std::string>> NeuralStereoMatcher::getAvailableModels() {
    return {
        {ModelType::STEREONET, "StereoNet - Fast stereo matching"},
        {ModelType::PSM_NET, "PSMNet - High quality stereo matching"},
        {ModelType::GA_NET, "GANet - Guided aggregation network"},
        {ModelType::HITNET, "HITNet - Hierarchical iterative tile refinement"}
    };
}

std::vector<InferenceStats> NeuralStereoMatcher::benchmarkModels(
    const std::vector<ModelType>& models, 
    const cv::Mat& test_left, 
    const cv::Mat& test_right) {
    
    std::vector<InferenceStats> results;
    
    for (auto model_type : models) {
        ModelConfig test_config;
        test_config.model_type = model_type;
        
        NeuralStereoMatcher matcher(test_config);
        matcher.initialize(test_config);
        
        // Run multiple iterations for benchmarking
        auto start = std::chrono::high_resolution_clock::now();
        const int iterations = 10;
        
        for (int i = 0; i < iterations; ++i) {
            cv::Mat result = matcher.computeDisparity(test_left, test_right);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        InferenceStats stats;
        stats.avg_fps = (iterations * 1000.0) / duration.count();
        stats.peak_fps = stats.avg_fps * 1.2; // Simulated peak
        stats.total_frames = iterations;
        stats.memory_usage_mb = 150.0; // Simulated
        
        results.push_back(stats);
    }
    
    return results;
}

Backend NeuralStereoMatcher::selectOptimalBackend() {
    auto available = getAvailableBackends();
    
    // Prefer GPU if available
    for (auto backend : available) {
        if (backend == Backend::ONNX_GPU) {
            return backend;
        }
    }
    
    return Backend::ONNX_CPU;
}

cv::Mat NeuralStereoMatcher::preprocessImages(const cv::Mat& left, const cv::Mat& right) {
    cv::Mat left_resized, right_resized;
    cv::resize(left, left_resized, cv::Size(config_.input_width, config_.input_height));
    cv::resize(right, right_resized, cv::Size(config_.input_width, config_.input_height));
    
    // Normalize
    left_resized.convertTo(left_resized, CV_32F, 1.0/255.0);
    right_resized.convertTo(right_resized, CV_32F, 1.0/255.0);
    
    // Concatenate channels for network input simulation
    std::vector<cv::Mat> channels = {left_resized, right_resized};
    cv::Mat concatenated;
    cv::merge(channels, concatenated);
    
    return concatenated;
}

cv::Mat NeuralStereoMatcher::postprocessDisparity(const std::vector<float>& raw_output) {
    cv::Mat disparity(config_.input_height, config_.input_width, CV_32F, (void*)raw_output.data());
    
    // Apply post-processing filters
    cv::medianBlur(disparity, disparity, 5);
    
    return disparity;
}

void NeuralStereoMatcher::updateStats(const std::chrono::high_resolution_clock::time_point& start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double current_fps = 1000.0 / duration.count();
    current_stats_.total_frames++;
    current_stats_.avg_fps = (current_stats_.avg_fps * (current_stats_.total_frames - 1) + current_fps) / current_stats_.total_frames;
    current_stats_.peak_fps = std::max(current_stats_.peak_fps, current_fps);
}

// Factory implementations
std::unique_ptr<NeuralStereoMatcher> NeuralMatcherFactory::createOptimalMatcher(const cv::Size& input_size) {
    ModelConfig config;
    config.input_width = input_size.width;
    config.input_height = input_size.height;
    config.model_type = ModelType::STEREONET;
    config.preferred_backend = Backend::AUTO;
    
    return std::make_unique<NeuralStereoMatcher>(config);
}

std::unique_ptr<NeuralStereoMatcher> NeuralMatcherFactory::createRealtimeMatcher() {
    ModelConfig config;
    config.input_width = 320;
    config.input_height = 240;
    config.model_type = ModelType::STEREONET;
    config.max_disparity = 64;
    
    return std::make_unique<NeuralStereoMatcher>(config);
}

std::unique_ptr<NeuralStereoMatcher> NeuralMatcherFactory::createHighQualityMatcher() {
    ModelConfig config;
    config.input_width = 1280;
    config.input_height = 720;
    config.model_type = ModelType::PSM_NET;
    config.max_disparity = 192;
    
    return std::make_unique<NeuralStereoMatcher>(config);
}

std::vector<std::pair<ModelType, std::string>> NeuralMatcherFactory::getAvailableModels() {
    return NeuralStereoMatcher::getAvailableModels();
}

// Adaptive Matcher implementations
AdaptiveNeuralMatcher::AdaptiveNeuralMatcher(const AdaptiveConfig& config) 
    : config_(config), current_matcher_index_(0) {
    
    // Create multiple matchers with different quality/speed tradeoffs
    matchers_.push_back(NeuralMatcherFactory::createRealtimeMatcher());
    matchers_.push_back(NeuralMatcherFactory::createOptimalMatcher());
    matchers_.push_back(NeuralMatcherFactory::createHighQualityMatcher());
    
    // Initialize all matchers
    for (auto& matcher : matchers_) {
        ModelConfig default_config;
        matcher->initialize(default_config);
    }
}

cv::Mat AdaptiveNeuralMatcher::processAdaptive(const cv::Mat& left, const cv::Mat& right) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto& current_matcher = matchers_[current_matcher_index_];
    cv::Mat result = current_matcher->computeDisparity(left, right);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double fps = 1000.0 / duration.count();
    double quality = 0.85; // Simulated quality metric
    
    updatePerformanceHistory(fps, quality);
    adaptConfiguration();
    
    return result;
}

void AdaptiveNeuralMatcher::updatePerformanceHistory(double fps, double quality) {
    recent_fps_.push_back(fps);
    recent_quality_.push_back(quality);
    
    // Keep only recent history
    if (recent_fps_.size() > config_.history_size) {
        recent_fps_.erase(recent_fps_.begin());
        recent_quality_.erase(recent_quality_.begin());
    }
    
    adaptive_state_.current_fps = fps;
    adaptive_state_.current_quality = quality;
    adaptive_state_.active_matcher_index = current_matcher_index_;
}

void AdaptiveNeuralMatcher::adaptConfiguration() {
    if (recent_fps_.size() < 3) return; // Need some history
    
    double avg_fps = std::accumulate(recent_fps_.begin(), recent_fps_.end(), 0.0) / recent_fps_.size();
    double avg_quality = std::accumulate(recent_quality_.begin(), recent_quality_.end(), 0.0) / recent_quality_.size();
    
    // Switch to faster model if FPS is too low
    if (avg_fps < config_.target_fps * 0.8 && current_matcher_index_ > 0) {
        current_matcher_index_--;
        std::cout << "Switching to faster model (index " << current_matcher_index_ << ")\n";
    }
    // Switch to higher quality model if FPS allows
    else if (avg_fps > config_.target_fps * 1.2 && 
             current_matcher_index_ < matchers_.size() - 1) {
        current_matcher_index_++;
        std::cout << "Switching to higher quality model (index " << current_matcher_index_ << ")\n";
    }
}

} // namespace ai
} // namespace stereovision
