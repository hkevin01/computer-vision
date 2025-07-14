#include "ai/enhanced_neural_matcher.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace stereovision::ai;

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void testBackendDetection() {
    printSeparator("🔍 Backend Detection Test");
    
    auto available_backends = EnhancedNeuralMatcher::getAvailableBackends();
    
    std::cout << "Available backends:" << std::endl;
    for (auto backend : available_backends) {
        std::string backend_name;
        switch (backend) {
            case EnhancedNeuralMatcher::Backend::ONNX_CPU:
                backend_name = "ONNX Runtime (CPU)";
                break;
            case EnhancedNeuralMatcher::Backend::ONNX_GPU:
                backend_name = "ONNX Runtime (GPU)";
                break;
            case EnhancedNeuralMatcher::Backend::TENSORRT:
                backend_name = "TensorRT";
                break;
            case EnhancedNeuralMatcher::Backend::OPENVINO:
                backend_name = "Intel OpenVINO";
                break;
            default:
                backend_name = "Unknown";
        }
        std::cout << "  ✅ " << backend_name << std::endl;
    }
    
    if (available_backends.empty()) {
        std::cout << "  ❌ No neural network backends available!" << std::endl;
        std::cout << "  📋 Install ONNX Runtime: sudo apt install libonnxruntime-dev" << std::endl;
    }
    
    auto optimal_backend = EnhancedNeuralMatcher::selectOptimalBackend();
    std::cout << "\nOptimal backend selected: ";
    switch (optimal_backend) {
        case EnhancedNeuralMatcher::Backend::ONNX_CPU:
            std::cout << "ONNX CPU";
            break;
        case EnhancedNeuralMatcher::Backend::ONNX_GPU:
            std::cout << "ONNX GPU";
            break;
        case EnhancedNeuralMatcher::Backend::TENSORRT:
            std::cout << "TensorRT";
            break;
        case EnhancedNeuralMatcher::Backend::OPENVINO:
            std::cout << "OpenVINO";
            break;
        default:
            std::cout << "AUTO/Unknown";
    }
    std::cout << std::endl;
}

void testModelInfo() {
    printSeparator("📋 Model Information Test");
    
    auto model_info = EnhancedNeuralMatcher::getModelInfo();
    
    std::cout << "Available neural network models:" << std::endl;
    for (const auto& [type, description] : model_info) {
        std::cout << "  🧠 " << description << std::endl;
    }
    
    // Test model manager
    ModelManager manager("models");
    auto available_models = manager.getAvailableModels();
    
    std::cout << "\nModel download information:" << std::endl;
    for (const auto& model : available_models) {
        std::cout << "  📥 " << model.name << " (" << model.file_size_mb << " MB)" << std::endl;
        std::cout << "      Input: " << model.input_size.width << "x" << model.input_size.height << std::endl;
        std::cout << "      Performance: ~" << model.fps_estimate << " FPS, Accuracy: " << model.accuracy_score << std::endl;
    }
}

void testMatcherCreation() {
    printSeparator("🔧 Matcher Creation Test");
    
    // Test factory methods
    std::cout << "Testing factory methods..." << std::endl;
    
    try {
        auto realtime_matcher = EnhancedMatcherFactory::createRealtimeMatcher();
        std::cout << "  ✅ Real-time matcher created successfully" << std::endl;
        
        auto quality_matcher = EnhancedMatcherFactory::createHighQualityMatcher();
        std::cout << "  ✅ High-quality matcher created successfully" << std::endl;
        
        auto optimal_matcher = EnhancedMatcherFactory::createOptimalMatcher(30.0, 0.8);
        std::cout << "  ✅ Optimal matcher created successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "  ❌ Factory method failed: " << e.what() << std::endl;
    }
}

void testBasicInference() {
    printSeparator("🧪 Basic Inference Test");
    
    // Create test images
    cv::Mat left_image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::Mat right_image = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // Add some pattern to make it interesting
    cv::rectangle(left_image, cv::Rect(100, 100, 200, 200), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(right_image, cv::Rect(90, 100, 200, 200), cv::Scalar(255, 255, 255), -1); // Slight horizontal shift
    
    cv::circle(left_image, cv::Point(400, 300), 50, cv::Scalar(128, 128, 255), -1);
    cv::circle(right_image, cv::Point(390, 300), 50, cv::Scalar(128, 128, 255), -1);
    
    std::cout << "Created test stereo pair: " << left_image.size() << std::endl;
    
    // Test with available backends
    auto available_backends = EnhancedNeuralMatcher::getAvailableBackends();
    
    if (available_backends.empty()) {
        std::cout << "  ⚠️ No backends available - skipping inference test" << std::endl;
        std::cout << "  📋 Install dependencies:" << std::endl;
        std::cout << "      sudo apt install libonnxruntime-dev" << std::endl;
        std::cout << "      python tools/model_manager.py download hitnet_kitti" << std::endl;
        return;
    }
    
    // Try to create a matcher (will use fallback if no models available)
    EnhancedNeuralMatcher::ModelConfig config;
    config.type = EnhancedNeuralMatcher::ModelType::HITNET;
    config.preferred_backend = available_backends[0];
    config.model_path = "models/hitnet_kitti.onnx"; // This may not exist yet
    
    EnhancedNeuralMatcher matcher(config);
    
    if (!matcher.initialize(config)) {
        std::cout << "  ⚠️ Model initialization failed (expected if model not downloaded)" << std::endl;
        std::cout << "  📋 Download models with: python tools/model_manager.py download-all" << std::endl;
        return;
    }
    
    std::cout << "  ✅ Matcher initialized successfully" << std::endl;
    
    // Perform inference
    auto start_time = std::chrono::high_resolution_clock::now();
    cv::Mat disparity = matcher.computeDisparity(left_image, right_image);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!disparity.empty()) {
        std::cout << "  ✅ Disparity computed successfully" << std::endl;
        std::cout << "      Size: " << disparity.size() << std::endl;
        std::cout << "      Type: " << disparity.type() << std::endl;
        std::cout << "      Time: " << duration.count() << " ms" << std::endl;
        
        // Print statistics
        auto stats = matcher.getLastStats();
        std::cout << "      Performance:" << std::endl;
        std::cout << "        FPS: " << stats.fps << std::endl;
        std::cout << "        Memory: " << stats.memory_usage_mb << " MB" << std::endl;
        std::cout << "        Backend: " << stats.backend_used << std::endl;
        
        // Basic disparity analysis
        double min_val, max_val;
        cv::minMaxLoc(disparity, &min_val, &max_val);
        cv::Scalar mean_val = cv::mean(disparity);
        
        std::cout << "      Disparity range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "      Mean disparity: " << mean_val[0] << std::endl;
        
    } else {
        std::cout << "  ❌ Disparity computation failed" << std::endl;
    }
}

void testConfidenceEstimation() {
    printSeparator("🎯 Confidence Estimation Test");
    
    // Similar to basic inference test but with confidence
    cv::Mat left_image = cv::Mat::zeros(240, 320, CV_8UC3);  // Smaller for speed
    cv::Mat right_image = cv::Mat::zeros(240, 320, CV_8UC3);
    
    // Add test patterns
    cv::rectangle(left_image, cv::Rect(50, 50, 100, 100), cv::Scalar(200, 200, 200), -1);
    cv::rectangle(right_image, cv::Rect(45, 50, 100, 100), cv::Scalar(200, 200, 200), -1);
    
    auto available_backends = EnhancedNeuralMatcher::getAvailableBackends();
    if (available_backends.empty()) {
        std::cout << "  ⚠️ No backends available - skipping confidence test" << std::endl;
        return;
    }
    
    EnhancedNeuralMatcher::ModelConfig config;
    config.preferred_backend = available_backends[0];
    config.model_path = "models/hitnet_kitti.onnx";
    
    EnhancedNeuralMatcher matcher(config);
    
    if (!matcher.initialize(config)) {
        std::cout << "  ⚠️ Model not available - skipping confidence test" << std::endl;
        return;
    }
    
    cv::Mat confidence;
    cv::Mat disparity = matcher.computeDisparityWithConfidence(left_image, right_image, confidence);
    
    if (!disparity.empty() && !confidence.empty()) {
        std::cout << "  ✅ Confidence estimation successful" << std::endl;
        
        auto quality_metrics = matcher.getLastQualityMetrics();
        std::cout << "      Quality metrics:" << std::endl;
        std::cout << "        Density: " << quality_metrics.density * 100 << "%" << std::endl;
        std::cout << "        Smoothness: " << quality_metrics.smoothness << std::endl;
        std::cout << "        Edge preservation: " << quality_metrics.edge_preservation << std::endl;
        std::cout << "        Confidence (mean±std): " << quality_metrics.confidence_mean 
                  << "±" << quality_metrics.confidence_std << std::endl;
    } else {
        std::cout << "  ❌ Confidence estimation failed" << std::endl;
    }
}

void printSystemInfo() {
    printSeparator("💻 System Information");
    
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Check for GPU support
    std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
    
    // Check for OpenCL
    std::cout << "OpenCL available: " << (cv::ocl::haveOpenCL() ? "Yes" : "No") << std::endl;
    
    // CPU info
    std::cout << "CPU threads: " << cv::getNumThreads() << std::endl;
    
    // Build configuration
    std::cout << "Build configuration:" << std::endl;
#ifdef WITH_ONNX
    std::cout << "  ✅ ONNX Runtime support" << std::endl;
#else
    std::cout << "  ❌ ONNX Runtime support" << std::endl;
#endif

#ifdef WITH_TENSORRT
    std::cout << "  ✅ TensorRT support" << std::endl;
#else
    std::cout << "  ❌ TensorRT support" << std::endl;
#endif

#ifdef WITH_OPENVINO
    std::cout << "  ✅ OpenVINO support" << std::endl;
#else
    std::cout << "  ❌ OpenVINO support" << std::endl;
#endif

#ifdef WITH_OPENCV_XIMGPROC
    std::cout << "  ✅ OpenCV ximgproc support" << std::endl;
#else
    std::cout << "  ❌ OpenCV ximgproc support" << std::endl;
#endif
}

int main(int argc, char* argv[]) {
    std::cout << "🧠 Enhanced Neural Stereo Matcher Test Suite" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        printSystemInfo();
        testBackendDetection();
        testModelInfo();
        testMatcherCreation();
        testBasicInference();
        testConfidenceEstimation();
        
        printSeparator("🎉 Test Summary");
        std::cout << "All tests completed!" << std::endl;
        std::cout << "\nNext steps:" << std::endl;
        std::cout << "1. Install ONNX Runtime: sudo apt install libonnxruntime-dev" << std::endl;
        std::cout << "2. Download models: python tools/model_manager.py download-all" << std::endl;
        std::cout << "3. Rebuild project: ./run.sh --clean" << std::endl;
        std::cout << "4. Test with real models: ./build/test_enhanced_neural_matcher" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
