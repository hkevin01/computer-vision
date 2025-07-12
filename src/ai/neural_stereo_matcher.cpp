#include "ai/neural_stereo_matcher.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>

namespace stereo_vision::ai {

// Implementation class to hide complex dependencies
class NeuralStereoMatcher::Impl {
public:
#ifdef WITH_TENSORRT
    std::unique_ptr<nvinfer1::IRuntime> trt_runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
    void* gpu_buffers_[4] = {nullptr}; // input_left, input_right, output_disp, output_conf
    cudaStream_t cuda_stream_;
#endif

#ifdef WITH_ONNX
    std::unique_ptr<Ort::Session> onnx_session_;
    Ort::Env onnx_env_{ORT_LOGGING_LEVEL_WARNING, "StereoMatcher"};
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)};
#endif

    Backend active_backend_ = Backend::AUTO;
    cv::Size input_size_;
    cv::Size original_size_;
    
    bool initializeTensorRT(const ModelConfig& config);
    bool initializeONNX(const ModelConfig& config);
    cv::Mat inferTensorRT(const cv::Mat& left, const cv::Mat& right);
    cv::Mat inferONNX(const cv::Mat& left, const cv::Mat& right);
};

NeuralStereoMatcher::NeuralStereoMatcher(const ModelConfig& config)
    : pImpl_(std::make_unique<Impl>()), config_(config) {
    if (!config.model_path.empty()) {
        initialize(config);
    }
}

NeuralStereoMatcher::~NeuralStereoMatcher() = default;

bool NeuralStereoMatcher::initialize(const ModelConfig& config) {
    config_ = config;
    
    // Select backend if AUTO
    if (config_.preferred_backend == Backend::AUTO) {
        config_.preferred_backend = selectOptimalBackend();
    }
    
    switch (config_.preferred_backend) {
        case Backend::TENSORRT:
#ifdef WITH_TENSORRT
            initialized_ = pImpl_->initializeTensorRT(config_);
            break;
#else
            std::cerr << "TensorRT not available, falling back to ONNX\n";
            [[fallthrough]];
#endif
        case Backend::ONNX_GPU:
        case Backend::ONNX_CPU:
#ifdef WITH_ONNX
            initialized_ = pImpl_->initializeONNX(config_);
            break;
#else
            std::cerr << "ONNX Runtime not available\n";
            return false;
#endif
        default:
            return false;
    }
    
    return initialized_;
}

cv::Mat NeuralStereoMatcher::computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image) {
    if (!initialized_) {
        std::cerr << "NeuralStereoMatcher not initialized\n";
        return cv::Mat();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cv::Mat result;
    switch (pImpl_->active_backend_) {
        case Backend::TENSORRT:
#ifdef WITH_TENSORRT
            result = pImpl_->inferTensorRT(left_image, right_image);
            break;
#endif
        case Backend::ONNX_GPU:
        case Backend::ONNX_CPU:
#ifdef WITH_ONNX
            result = pImpl_->inferONNX(left_image, right_image);
            break;
#endif
        default:
            break;
    }
    
    updateStats(start_time);
    return result;
}

cv::Mat NeuralStereoMatcher::computeDisparityWithConfidence(const cv::Mat& left_image, 
                                                          const cv::Mat& right_image,
                                                          cv::Mat& confidence) {
    // For now, compute disparity and generate confidence from disparity consistency
    cv::Mat disparity = computeDisparity(left_image, right_image);
    
    if (disparity.empty()) {
        confidence = cv::Mat();
        return disparity;
    }
    
    // Simple confidence estimation based on disparity gradient
    cv::Mat grad_x, grad_y;
    cv::Sobel(disparity, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(disparity, grad_y, CV_32F, 0, 1, 3);
    
    cv::Mat gradient_magnitude;
    cv::magnitude(grad_x, grad_y, gradient_magnitude);
    
    // Higher gradients = lower confidence
    cv::normalize(gradient_magnitude, confidence, 0, 1, cv::NORM_MINMAX);
    confidence = 1.0 - confidence;
    
    return disparity;
}

std::vector<NeuralStereoMatcher::Backend> NeuralStereoMatcher::getAvailableBackends() {
    std::vector<Backend> backends;
    
#ifdef WITH_TENSORRT
    backends.push_back(Backend::TENSORRT);
#endif
#ifdef WITH_ONNX
    backends.push_back(Backend::ONNX_GPU);
    backends.push_back(Backend::ONNX_CPU);
#endif
    
    return backends;
}

std::string NeuralStereoMatcher::downloadPretrainedModel(ModelType model_type, 
                                                       const std::string& output_dir) {
    // Model URLs (these would be real URLs in production)
    std::map<ModelType, std::string> model_urls = {
        {ModelType::HITNET, "https://github.com/google-research/google-research/releases/download/hitnet/hitnet_flyingthings_finalpass_xl.pb"},
        {ModelType::RAFT_STEREO, "https://github.com/princeton-vl/RAFT-Stereo/releases/download/models/raftstereo-middlebury.pth"},
        {ModelType::STTR, "https://github.com/mli0603/stereo-transformer/releases/download/v1.0/sttr_light_sceneflow_pretrained_model.pth.tar"}
    };
    
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    // In a real implementation, this would download the model
    // For now, return a placeholder path
    std::string model_name;
    switch (model_type) {
        case ModelType::HITNET: model_name = "hitnet.onnx"; break;
        case ModelType::RAFT_STEREO: model_name = "raft_stereo.onnx"; break;
        case ModelType::STTR: model_name = "sttr.onnx"; break;
        default: model_name = "model.onnx"; break;
    }
    
    return output_dir + "/" + model_name;
}

std::map<std::string, NeuralStereoMatcher::InferenceStats> NeuralStereoMatcher::benchmarkModels(
    const std::vector<std::pair<cv::Mat, cv::Mat>>& test_images) {
    
    std::map<std::string, InferenceStats> results;
    
    auto backends = getAvailableBackends();
    std::vector<ModelType> models = {ModelType::HITNET, ModelType::RAFT_STEREO, ModelType::STTR};
    
    for (auto backend : backends) {
        for (auto model : models) {
            ModelConfig config;
            config.type = model;
            config.preferred_backend = backend;
            config.model_path = downloadPretrainedModel(model);
            
            NeuralStereoMatcher matcher(config);
            if (!matcher.isInitialized()) continue;
            
            std::string key = std::to_string(static_cast<int>(model)) + "_" + 
                            std::to_string(static_cast<int>(backend));
            
            InferenceStats total_stats{};
            for (const auto& image_pair : test_images) {
                matcher.computeDisparity(image_pair.first, image_pair.second);
                auto stats = matcher.getLastStats();
                total_stats.inference_time_ms += stats.inference_time_ms;
                total_stats.total_time_ms += stats.total_time_ms;
            }
            
            total_stats.inference_time_ms /= test_images.size();
            total_stats.total_time_ms /= test_images.size();
            total_stats.fps = 1000.0 / total_stats.total_time_ms;
            
            results[key] = total_stats;
        }
    }
    
    return results;
}

// Private implementation methods
cv::Mat NeuralStereoMatcher::preprocessImages(const cv::Mat& left, const cv::Mat& right) {
    cv::Mat left_resized, right_resized;
    cv::resize(left, left_resized, cv::Size(config_.input_width, config_.input_height));
    cv::resize(right, right_resized, cv::Size(config_.input_width, config_.input_height));
    
    // Normalize to [0, 1]
    left_resized.convertTo(left_resized, CV_32F, 1.0/255.0);
    right_resized.convertTo(right_resized, CV_32F, 1.0/255.0);
    
    // Concatenate along channel dimension for some models
    cv::Mat combined;
    std::vector<cv::Mat> channels = {left_resized, right_resized};
    cv::merge(channels, combined);
    
    return combined;
}

cv::Mat NeuralStereoMatcher::postprocessDisparity(const std::vector<float>& raw_output) {
    cv::Mat disparity(config_.input_height, config_.input_width, CV_32F, (void*)raw_output.data());
    
    // Scale disparity values
    disparity *= config_.max_disparity;
    
    return disparity.clone();
}

NeuralStereoMatcher::Backend NeuralStereoMatcher::selectOptimalBackend() {
#ifdef WITH_TENSORRT
    return Backend::TENSORRT;
#elif defined(WITH_ONNX)
    return Backend::ONNX_GPU;
#else
    return Backend::ONNX_CPU;
#endif
}

void NeuralStereoMatcher::updateStats(const std::chrono::high_resolution_clock::time_point& start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    last_stats_.total_time_ms = duration.count() / 1000.0;
    last_stats_.fps = 1000.0 / last_stats_.total_time_ms;
    last_stats_.backend_used = std::to_string(static_cast<int>(pImpl_->active_backend_));
}

// TensorRT implementation (when available)
#ifdef WITH_TENSORRT
bool NeuralStereoMatcher::Impl::initializeTensorRT(const ModelConfig& config) {
    // Implementation would load TensorRT engine from file
    // This is a placeholder for the complex TensorRT initialization
    return false; // Not implemented in this example
}

cv::Mat NeuralStereoMatcher::Impl::inferTensorRT(const cv::Mat& left, const cv::Mat& right) {
    // TensorRT inference implementation
    return cv::Mat();
}
#endif

// ONNX implementation (when available)
#ifdef WITH_ONNX
bool NeuralStereoMatcher::Impl::initializeONNX(const ModelConfig& config) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        if (config.preferred_backend == Backend::ONNX_GPU) {
            // Enable GPU provider if available
            OrtCUDAProviderOptions cuda_options{};
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        
        onnx_session_ = std::make_unique<Ort::Session>(onnx_env_, config.model_path.c_str(), session_options);
        active_backend_ = config.preferred_backend;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNX initialization failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat NeuralStereoMatcher::Impl::inferONNX(const cv::Mat& left, const cv::Mat& right) {
    // ONNX inference implementation
    // This is a simplified version - real implementation would be more complex
    return cv::Mat();
}
#endif

// Factory implementations
std::unique_ptr<NeuralStereoMatcher> NeuralMatcherFactory::createOptimalMatcher(
    double performance_target_fps, double quality_target) {
    
    ModelConfig config;
    
    if (performance_target_fps > 60) {
        config.type = ModelType::HITNET;
        config.use_fp16 = true;
    } else if (quality_target > 0.9) {
        config.type = ModelType::RAFT_STEREO;
        config.use_fp16 = false;
    } else {
        config.type = ModelType::STTR;
        config.use_fp16 = true;
    }
    
    config.model_path = NeuralStereoMatcher::downloadPretrainedModel(config.type);
    
    return std::make_unique<NeuralStereoMatcher>(config);
}

std::unique_ptr<NeuralStereoMatcher> NeuralMatcherFactory::createRealtimeMatcher() {
    return createOptimalMatcher(60.0, 0.7);
}

std::unique_ptr<NeuralStereoMatcher> NeuralMatcherFactory::createHighQualityMatcher() {
    return createOptimalMatcher(15.0, 0.95);
}

std::vector<std::pair<NeuralStereoMatcher::ModelType, std::string>> NeuralMatcherFactory::getAvailableModels() {
    return {
        {NeuralStereoMatcher::ModelType::HITNET, "HITNet - Real-time stereo matching"},
        {NeuralStereoMatcher::ModelType::RAFT_STEREO, "RAFT-Stereo - High accuracy stereo"},
        {NeuralStereoMatcher::ModelType::STTR, "Stereo Transformer - Balanced performance"},
    };
}

// Adaptive matcher implementation
AdaptiveNeuralMatcher::AdaptiveNeuralMatcher(const AdaptiveConfig& config)
    : config_(config) {
    
    // Create multiple matchers with different configurations
    matchers_.push_back(NeuralMatcherFactory::createRealtimeMatcher());
    matchers_.push_back(NeuralMatcherFactory::createOptimalMatcher(30.0, 0.8));
    matchers_.push_back(NeuralMatcherFactory::createHighQualityMatcher());
}

cv::Mat AdaptiveNeuralMatcher::processAdaptive(const cv::Mat& left, const cv::Mat& right) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto& current_matcher = matchers_[current_matcher_index_];
    cv::Mat result = current_matcher->computeDisparity(left, right);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double fps = 1000.0 / duration.count();
    
    // Simple quality estimation (in practice, would use more sophisticated metrics)
    double quality = 0.8; // Placeholder
    
    updatePerformanceHistory(fps, quality);
    adaptConfiguration();
    
    return result;
}

AdaptiveNeuralMatcher::AdaptiveState AdaptiveNeuralMatcher::getAdaptiveState() const {
    AdaptiveState state;
    state.current_fps = recent_fps_.empty() ? 0.0 : recent_fps_.back();
    state.current_quality = recent_quality_.empty() ? 0.0 : recent_quality_.back();
    state.current_model = matchers_[current_matcher_index_]->getConfig().type;
    state.current_resolution = cv::Size(
        matchers_[current_matcher_index_]->getConfig().input_width,
        matchers_[current_matcher_index_]->getConfig().input_height);
    state.is_adapting = recent_fps_.size() > 1;
    
    return state;
}

void AdaptiveNeuralMatcher::updatePerformanceHistory(double fps, double quality) {
    recent_fps_.push_back(fps);
    recent_quality_.push_back(quality);
    
    // Keep only recent history
    const size_t max_history = 10;
    if (recent_fps_.size() > max_history) {
        recent_fps_.erase(recent_fps_.begin());
        recent_quality_.erase(recent_quality_.begin());
    }
}

void AdaptiveNeuralMatcher::adaptConfiguration() {
    if (recent_fps_.size() < 3) return; // Need some history
    
    double avg_fps = std::accumulate(recent_fps_.begin(), recent_fps_.end(), 0.0) / recent_fps_.size();
    double avg_quality = std::accumulate(recent_quality_.begin(), recent_quality_.end(), 0.0) / recent_quality_.size();
    
    // Adapt based on performance
    if (avg_fps < config_.target_fps * 0.8 && current_matcher_index_ > 0) {
        // Switch to faster model
        current_matcher_index_--;
    } else if (avg_fps > config_.target_fps * 1.2 && avg_quality < config_.max_quality && 
               current_matcher_index_ < matchers_.size() - 1) {
        // Switch to higher quality model
        current_matcher_index_++;
    }
}

} // namespace stereo_vision::ai
