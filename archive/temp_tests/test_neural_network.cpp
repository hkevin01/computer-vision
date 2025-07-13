// Test program for Priority 2 Neural Network features
#include "include/ai/neural_stereo_matcher_simple.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace stereovision::ai;

int main() {
    std::cout << "🧠 Testing Neural Stereo Matching Features" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        // Test basic neural matcher
        std::cout << "\n1. Testing Basic Neural Matcher..." << std::endl;
        ModelConfig config;
        config.model_type = ModelType::STEREONET;
        config.preferred_backend = Backend::AUTO;
        
        NeuralStereoMatcher matcher(config);
        
        if (matcher.initialize(config)) {
            std::cout << "✅ Neural matcher initialized successfully" << std::endl;
        } else {
            std::cout << "❌ Failed to initialize neural matcher" << std::endl;
            return 1;
        }
        
        // Create test images
        cv::Mat left_image = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat right_image = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Add some random content
        cv::randu(left_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::randu(right_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        
        std::cout << "\n2. Testing Disparity Computation..." << std::endl;
        cv::Mat disparity = matcher.computeDisparity(left_image, right_image);
        
        if (!disparity.empty()) {
            std::cout << "✅ Disparity computation successful" << std::endl;
            std::cout << "   Disparity map size: " << disparity.size() << std::endl;
        } else {
            std::cout << "❌ Disparity computation failed" << std::endl;
            return 1;
        }
        
        // Test confidence map
        std::cout << "\n3. Testing Confidence Map..." << std::endl;
        cv::Mat confidence_map;
        cv::Mat disparity_with_confidence = matcher.computeDisparityWithConfidence(
            left_image, right_image, confidence_map);
        
        if (!confidence_map.empty()) {
            std::cout << "✅ Confidence map generation successful" << std::endl;
            std::cout << "   Confidence map size: " << confidence_map.size() << std::endl;
        } else {
            std::cout << "❌ Confidence map generation failed" << std::endl;
        }
        
        // Test available backends
        std::cout << "\n4. Testing Backend Detection..." << std::endl;
        auto backends = NeuralStereoMatcher::getAvailableBackends();
        std::cout << "Available backends: " << backends.size() << std::endl;
        for (size_t i = 0; i < backends.size(); ++i) {
            std::cout << "   Backend " << i << ": " << static_cast<int>(backends[i]) << std::endl;
        }
        
        // Test model information
        std::cout << "\n5. Testing Model Information..." << std::endl;
        auto models = NeuralStereoMatcher::getAvailableModels();
        std::cout << "Available models: " << models.size() << std::endl;
        for (const auto& model : models) {
            std::cout << "   " << model.second << std::endl;
        }
        
        // Test factory methods
        std::cout << "\n6. Testing Factory Methods..." << std::endl;
        auto realtime_matcher = NeuralMatcherFactory::createRealtimeMatcher();
        auto quality_matcher = NeuralMatcherFactory::createHighQualityMatcher();
        auto optimal_matcher = NeuralMatcherFactory::createOptimalMatcher();
        
        if (realtime_matcher && quality_matcher && optimal_matcher) {
            std::cout << "✅ All factory methods working" << std::endl;
        } else {
            std::cout << "❌ Some factory methods failed" << std::endl;
        }
        
        // Test adaptive matcher
        std::cout << "\n7. Testing Adaptive Neural Matcher..." << std::endl;
        AdaptiveConfig adaptive_config;
        adaptive_config.target_fps = 30.0;
        adaptive_config.quality_threshold = 0.8;
        
        AdaptiveNeuralMatcher adaptive_matcher(adaptive_config);
        
        // Run several adaptive processing cycles
        for (int i = 0; i < 5; ++i) {
            cv::Mat adaptive_result = adaptive_matcher.processAdaptive(left_image, right_image);
            if (!adaptive_result.empty()) {
                auto state = adaptive_matcher.getAdaptiveState();
                std::cout << "   Cycle " << i << ": FPS=" << state.current_fps 
                         << ", Quality=" << state.current_quality 
                         << ", Matcher=" << state.active_matcher_index << std::endl;
            }
        }
        
        // Test benchmarking
        std::cout << "\n8. Testing Model Benchmarking..." << std::endl;
        std::vector<ModelType> models_to_benchmark = {
            ModelType::STEREONET,
            ModelType::PSM_NET
        };
        
        auto benchmark_results = matcher.benchmarkModels(models_to_benchmark, left_image, right_image);
        
        std::cout << "Benchmark results:" << std::endl;
        for (size_t i = 0; i < benchmark_results.size(); ++i) {
            const auto& result = benchmark_results[i];
            std::cout << "   Model " << i << ": " << result.avg_fps << " FPS, " 
                     << result.memory_usage_mb << " MB" << std::endl;
        }
        
        // Test statistics
        std::cout << "\n9. Testing Performance Statistics..." << std::endl;
        auto stats = matcher.getStats();
        std::cout << "Current stats:" << std::endl;
        std::cout << "   Average FPS: " << stats.avg_fps << std::endl;
        std::cout << "   Peak FPS: " << stats.peak_fps << std::endl;
        std::cout << "   Total frames: " << stats.total_frames << std::endl;
        std::cout << "   Memory usage: " << stats.memory_usage_mb << " MB" << std::endl;
        
        std::cout << "\n🎉 All Neural Network tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
