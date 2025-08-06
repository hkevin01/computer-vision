# Enhanced Neural Stereo Matcher Implementation Plan

## 🎯 Immediate Action Items (Week 1-2)

### 1. Real ONNX Model Integration

#### Current Issue
The neural stereo matcher is using OpenCV's StereoSGBM as a placeholder instead of actual neural networks.

#### Solution: Replace Placeholder with Real Models
```cpp
// File: include/ai/enhanced_neural_matcher.hpp
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace stereovision::ai {

class EnhancedNeuralMatcher {
private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // Model metadata
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // Processing parameters
    cv::Size model_input_size_;
    float disparity_scale_;
    
public:
    // Model initialization
    bool loadModel(const std::string& model_path, 
                   bool use_gpu = true, 
                   int num_threads = 4);
    
    // Core inference
    cv::Mat computeDisparity(const cv::Mat& left, const cv::Mat& right) override;
    
    // Batch processing for efficiency
    std::vector<cv::Mat> computeDisparityBatch(
        const std::vector<cv::Mat>& left_images,
        const std::vector<cv::Mat>& right_images);
    
private:
    // Image preprocessing for neural networks
    std::pair<cv::Mat, cv::Mat> preprocessStereoImages(
        const cv::Mat& left, const cv::Mat& right);
    
    // Convert OpenCV Mat to ONNX tensor
    Ort::Value matToTensor(const cv::Mat& mat, 
                          const std::vector<int64_t>& shape);
    
    // Convert ONNX tensor to OpenCV Mat
    cv::Mat tensorToMat(Ort::Value& tensor, 
                       const cv::Size& output_size);
};

} // namespace stereovision::ai
```

### 2. Download and Integrate Pre-trained Models

#### Model Selection Strategy
Create a model zoo with different performance/quality tradeoffs:

```cpp
// File: src/ai/model_zoo.cpp
#include "ai/model_zoo.hpp"
#include <filesystem>
#include <fstream>
#include <curl/curl.h>  // For downloading models

namespace stereovision::ai {

class ModelZoo {
public:
    struct ModelInfo {
        std::string name;
        std::string url;
        std::string filename;
        cv::Size input_size;
        float max_disparity;
        float fps_estimate;      // Expected FPS on RTX 3060
        float accuracy_score;    // Relative accuracy (0-1)
        size_t file_size_mb;
    };
    
    // Available models ranked by speed/quality
    static const std::vector<ModelInfo> AVAILABLE_MODELS;
    
    // Download model if not present
    static bool downloadModel(const ModelInfo& model, 
                             const std::string& models_dir = "models/");
    
    // Verify model integrity
    static bool verifyModel(const std::string& model_path);
    
    // Get optimal model for requirements
    static ModelInfo selectOptimalModel(float target_fps, 
                                       float min_accuracy = 0.7);
};

// Model definitions
const std::vector<ModelZoo::ModelInfo> ModelZoo::AVAILABLE_MODELS = {
    // Real-time models (>60 FPS)
    {
        "HITNet_KITTI_realtime",
        "https://github.com/google-research/google-research/releases/download/hitnet/hitnet_sf_finalpass_720x1280.onnx",
        "hitnet_realtime.onnx",
        cv::Size(1280, 720),
        192.0f,
        80.0f,      // 80 FPS
        0.75f,      // Good accuracy
        45          // 45 MB
    },
    // Balanced models (30-60 FPS)
    {
        "RAFT_Stereo_balanced", 
        "https://huggingface.co/models/raftstereo/resolve/main/raftstereo_middlebury.onnx",
        "raftstereo_balanced.onnx",
        cv::Size(640, 480),
        256.0f,
        45.0f,      // 45 FPS
        0.88f,      // High accuracy
        120         // 120 MB
    },
    // High-quality models (10-30 FPS)
    {
        "CREStereo_high_quality",
        "https://github.com/megvii-research/CREStereo/releases/download/v1.0/crestereo_combined_iter10.onnx",
        "crestereo_hq.onnx", 
        cv::Size(1024, 768),
        320.0f,
        25.0f,      // 25 FPS
        0.94f,      // Excellent accuracy
        250         // 250 MB
    }
};

} // namespace stereovision::ai
```

### 3. GPU Memory Management

#### Efficient Memory Pool
```cpp
// File: include/ai/gpu_memory_manager.hpp
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <queue>
#include <mutex>

namespace stereovision::gpu {

class GPUMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_used;
    };
    
    std::unordered_map<size_t, std::queue<MemoryBlock*>> free_blocks_;
    std::vector<std::unique_ptr<MemoryBlock>> all_blocks_;
    std::mutex pool_mutex_;
    size_t total_allocated_;
    size_t max_pool_size_;
    
public:
    explicit GPUMemoryPool(size_t max_size = 2ULL * 1024 * 1024 * 1024); // 2GB
    ~GPUMemoryPool();
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void cleanup(); // Free unused blocks
    
    // Statistics
    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getAvailableMemory() const;
    void printStatistics() const;
};

// Global memory pool instance
extern std::unique_ptr<GPUMemoryPool> g_gpu_memory_pool;

} // namespace stereovision::gpu
```

### 4. Benchmark Integration

#### Automated Model Benchmarking
```cpp
// File: include/ai/model_benchmark.hpp
#pragma once

#include "ai/enhanced_neural_matcher.hpp"
#include <chrono>
#include <vector>
#include <string>

namespace stereovision::ai {

struct BenchmarkResult {
    std::string model_name;
    float avg_fps;
    float min_fps;
    float max_fps;
    float memory_usage_mb;
    float accuracy_score;  // If ground truth available
    std::vector<float> frame_times_ms;
};

class ModelBenchmark {
private:
    std::vector<std::pair<cv::Mat, cv::Mat>> test_image_pairs_;
    std::vector<cv::Mat> ground_truth_disparities_;  // Optional
    
public:
    // Load test dataset
    bool loadTestData(const std::string& dataset_path);
    bool loadKITTIData(const std::string& kitti_path);
    bool loadMiddleburyData(const std::string& middlebury_path);
    
    // Benchmark single model
    BenchmarkResult benchmarkModel(EnhancedNeuralMatcher& matcher,
                                  int num_warmup_runs = 10,
                                  int num_benchmark_runs = 100);
    
    // Compare multiple models
    std::vector<BenchmarkResult> compareModels(
        const std::vector<std::string>& model_paths);
    
    // Generate benchmark report
    void generateReport(const std::vector<BenchmarkResult>& results,
                       const std::string& output_path = "benchmark_report.html");
    
private:
    // Quality metrics
    float computeDisparityError(const cv::Mat& predicted, 
                               const cv::Mat& ground_truth);
    float computeBadPixelPercentage(const cv::Mat& predicted, 
                                   const cv::Mat& ground_truth, 
                                   float threshold = 3.0f);
};

} // namespace stereovision::ai
```

### 5. Advanced Preprocessing Pipeline

#### Intelligent Image Enhancement
```cpp
// File: include/ai/image_preprocessor.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

namespace stereovision::preprocessing {

class IntelligentPreprocessor {
public:
    struct ProcessingParams {
        bool auto_exposure = true;
        bool noise_reduction = true;
        bool edge_enhancement = false;
        bool color_correction = true;
        float brightness_factor = 1.0f;
        float contrast_factor = 1.0f;
    };
    
    struct ImageAnalysis {
        float brightness_level;     // 0-1
        float contrast_level;       // 0-1  
        float noise_level;          // 0-1
        float edge_density;         // 0-1
        bool is_low_light;
        bool is_overexposed;
        bool has_motion_blur;
    };
    
public:
    // Analyze image characteristics
    ImageAnalysis analyzeImage(const cv::Mat& image);
    
    // Adaptive preprocessing based on analysis
    std::pair<cv::Mat, cv::Mat> adaptivePreprocess(
        const cv::Mat& left, const cv::Mat& right);
    
    // Specific enhancement methods
    cv::Mat enhanceLowLight(const cv::Mat& image);
    cv::Mat reduceNoise(const cv::Mat& image, float strength = 0.5f);
    cv::Mat enhanceEdges(const cv::Mat& image, float strength = 0.3f);
    cv::Mat correctColors(const cv::Mat& image);
    
    // Rectification quality assessment
    float assessRectificationQuality(const cv::Mat& left, const cv::Mat& right);
    
private:
    // Advanced denoising using neural networks
    cv::Mat neuralDenoise(const cv::Mat& image);
    
    // Histogram analysis
    void analyzeHistogram(const cv::Mat& image, ImageAnalysis& analysis);
    
    // Edge detection and analysis
    void analyzeEdges(const cv::Mat& image, ImageAnalysis& analysis);
};

} // namespace stereovision::preprocessing
```

## 🚀 Implementation Priority Order

### Week 1: Core Infrastructure
1. **Set up ONNX Runtime properly** - Replace placeholder implementations
2. **Create ModelZoo class** - Download and manage pre-trained models
3. **Implement EnhancedNeuralMatcher** - Real neural network inference

### Week 2: Performance Optimization  
1. **GPU Memory Pool** - Efficient memory management
2. **Batch Processing** - Process multiple image pairs efficiently
3. **Asynchronous Pipeline** - Non-blocking inference

### Week 3: Quality Improvements
1. **Intelligent Preprocessing** - Adaptive image enhancement
2. **Post-processing Pipeline** - Edge-preserving filtering
3. **Confidence Estimation** - Multiple confidence metrics

### Week 4: Testing & Validation
1. **Automated Benchmarking** - Compare models systematically  
2. **Quality Metrics** - Accuracy assessment against ground truth
3. **Integration Testing** - End-to-end testing with GUI

## 📋 Success Metrics

### Performance Targets
- [ ] **Real Inference**: Replace OpenCV placeholder with actual neural networks
- [ ] **Speed**: Achieve 60+ FPS on RTX 3060 with HITNet model
- [ ] **Quality**: <3% bad pixel rate on KITTI benchmark  
- [ ] **Memory**: Use <1GB GPU memory for real-time models
- [ ] **Robustness**: Handle various lighting conditions automatically

### User Experience Goals
- [ ] **Automatic Model Selection**: Choose optimal model based on hardware
- [ ] **Progressive Loading**: Start with fast model, upgrade to quality model  
- [ ] **Real-time Feedback**: Show processing quality metrics
- [ ] **Error Recovery**: Graceful fallback when models fail

This implementation plan focuses on the most impactful improvements that will transform your project from a functional prototype into a professional-grade stereo vision system.
