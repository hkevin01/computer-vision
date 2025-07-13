// Test program for Priority 2 Performance Benchmarking features
#include "include/benchmark/performance_benchmark_simple.hpp"
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>

using namespace stereovision::benchmark;

int main() {
    std::cout << "⚡ Testing Performance Benchmarking Features" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    try {
        // Test system information
        std::cout << "\n1. Testing System Information..." << std::endl;
        PerformanceBenchmark benchmark;
        auto system_info = benchmark.getSystemInfo();
        
        std::cout << "System Information:" << std::endl;
        std::cout << "   OS: " << system_info.os_name << std::endl;
        std::cout << "   CPU: " << system_info.cpu_model << " (" << system_info.cpu_cores << " cores)" << std::endl;
        std::cout << "   GPU: " << system_info.gpu_model << std::endl;
        std::cout << "   Memory: " << system_info.total_memory_mb << " MB" << std::endl;
        std::cout << "   OpenCV: " << system_info.opencv_version << std::endl;
        
        // Test basic benchmarking
        std::cout << "\n2. Testing Basic Performance Measurement..." << std::endl;
        double cpu_usage = benchmark.getCurrentCPUUsage();
        double memory_usage = benchmark.getCurrentMemoryUsage();
        double gpu_usage = benchmark.getCurrentGPUUsage();
        
        std::cout << "Current system metrics:" << std::endl;
        std::cout << "   CPU Usage: " << cpu_usage << "%" << std::endl;
        std::cout << "   Memory Usage: " << memory_usage << " MB" << std::endl;
        std::cout << "   GPU Usage: " << gpu_usage << "%" << std::endl;
        
        // Test stereo matching benchmark
        std::cout << "\n3. Testing Stereo Matching Benchmark..." << std::endl;
        
        // Create test images
        cv::Mat left_image = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat right_image = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::randu(left_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::randu(right_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        
        benchmark.setTestImages(left_image, right_image);
        benchmark.setTestDuration(3); // 3 seconds for quick test
        benchmark.setWarmupFrames(10);
        
        // Test OpenCV stereo algorithms
        auto stereo_bm_algorithm = [](const cv::Mat& left, const cv::Mat& right) -> cv::Mat {
            cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(64, 21);
            cv::Mat disparity, left_gray, right_gray;
            
            if (left.channels() == 3) {
                cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
            } else {
                left_gray = left; right_gray = right;
            }
            
            stereo->compute(left_gray, right_gray, disparity);
            return disparity;
        };
        
        auto stereo_sgbm_algorithm = [](const cv::Mat& left, const cv::Mat& right) -> cv::Mat {
            cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, 64, 21);
            cv::Mat disparity, left_gray, right_gray;
            
            if (left.channels() == 3) {
                cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
            } else {
                left_gray = left; right_gray = right;
            }
            
            stereo->compute(left_gray, right_gray, disparity);
            return disparity;
        };
        
        // Benchmark different algorithms
        auto bm_result = benchmark.benchmarkStereoMatching("StereoBM", stereo_bm_algorithm);
        auto sgbm_result = benchmark.benchmarkStereoMatching("StereoSGBM", stereo_sgbm_algorithm);
        
        std::cout << "Stereo Algorithm Results:" << std::endl;
        std::cout << "   StereoBM: " << bm_result.avg_fps << " FPS" << std::endl;
        std::cout << "   StereoSGBM: " << sgbm_result.avg_fps << " FPS" << std::endl;
        
        // Test neural network benchmarking
        std::cout << "\n4. Testing Neural Network Benchmarking..." << std::endl;
        auto neural_result1 = benchmark.benchmarkNeuralMatcher("StereoNet");
        auto neural_result2 = benchmark.benchmarkNeuralMatcher("PSMNet");
        
        std::cout << "Neural Network Results:" << std::endl;
        std::cout << "   StereoNet: " << neural_result1.avg_fps << " FPS" << std::endl;
        std::cout << "   PSMNet: " << neural_result2.avg_fps << " FPS" << std::endl;
        
        // Test multi-camera benchmarking
        std::cout << "\n5. Testing Multi-Camera Benchmarking..." << std::endl;
        auto multicam_2 = benchmark.benchmarkMultiCamera(2);
        auto multicam_4 = benchmark.benchmarkMultiCamera(4);
        
        std::cout << "Multi-Camera Results:" << std::endl;
        std::cout << "   2 cameras: " << multicam_2.avg_fps << " FPS" << std::endl;
        std::cout << "   4 cameras: " << multicam_4.avg_fps << " FPS" << std::endl;
        
        // Test full benchmark suite
        std::cout << "\n6. Testing Full Benchmark Suite..." << std::endl;
        auto suite_results = benchmark.runFullBenchmarkSuite();
        
        std::cout << "Full benchmark suite completed with " << suite_results.size() << " tests" << std::endl;
        benchmark.printSummary(suite_results);
        
        // Test report generation
        std::cout << "\n7. Testing Report Generation..." << std::endl;
        
        // Generate HTML report
        benchmark.generateHTMLReport(suite_results, "benchmark_report.html");
        std::cout << "✅ HTML report generated" << std::endl;
        
        // Generate CSV report
        benchmark.generateCSVReport(suite_results, "benchmark_results.csv");
        std::cout << "✅ CSV report generated" << std::endl;
        
        // Test baseline functionality
        benchmark.saveBaseline("performance_baseline.csv", suite_results);
        std::cout << "✅ Baseline saved" << std::endl;
        
        if (benchmark.runRegressionTest("performance_baseline.csv")) {
            std::cout << "✅ Regression test passed" << std::endl;
        }
        
        // Test real-time benchmarking
        std::cout << "\n8. Testing Real-time Benchmarking..." << std::endl;
        RealtimeBenchmark realtime_benchmark;
        
        // Set performance thresholds
        realtime_benchmark.setPerformanceThresholds(15.0, 100.0, 512.0); // 15 FPS, 100ms, 512MB
        
        realtime_benchmark.startMonitoring();
        
        // Simulate some frame processing
        for (int i = 0; i < 50; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Simulate processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(20 + (i % 30)));
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            
            realtime_benchmark.recordFrame(duration);
            realtime_benchmark.recordMemoryUsage(200.0 + (i % 100));
            realtime_benchmark.recordCPUUsage(30.0 + (i % 40));
        }
        
        std::cout << "Real-time metrics:" << std::endl;
        std::cout << "   Current FPS: " << realtime_benchmark.getCurrentFPS() << std::endl;
        std::cout << "   Average Latency: " << realtime_benchmark.getAverageLatency() << " ms" << std::endl;
        std::cout << "   Peak Memory: " << realtime_benchmark.getPeakMemoryUsage() << " MB" << std::endl;
        
        if (realtime_benchmark.hasPerformanceIssues()) {
            auto alerts = realtime_benchmark.getPerformanceAlerts();
            std::cout << "Performance alerts:" << std::endl;
            for (const auto& alert : alerts) {
                std::cout << "   ⚠️  " << alert << std::endl;
            }
        } else {
            std::cout << "✅ No performance issues detected" << std::endl;
        }
        
        realtime_benchmark.stopMonitoring();
        
        // Test benchmark suite static methods
        std::cout << "\n9. Testing Benchmark Suite..." << std::endl;
        
        auto quick_results = BenchmarkSuite::runQuickBenchmark();
        std::cout << "Quick benchmark: " << quick_results.size() << " tests completed" << std::endl;
        
        auto component_results = BenchmarkSuite::benchmarkStereoAlgorithms();
        std::cout << "Component benchmark: " << component_results.test_name 
                 << " - " << component_results.avg_fps << " FPS" << std::endl;
        
        // Test stress testing
        std::cout << "\n10. Testing Stress Testing (short version)..." << std::endl;
        benchmark.setTestDuration(2); // Very short for demo
        auto stress_results = benchmark.runStressTest(1); // 1 minute becomes 2 seconds due to override
        
        std::cout << "Stress test completed with " << stress_results.size() << " tests" << std::endl;
        
        // Test comparison report generation
        std::vector<std::vector<BenchmarkResult>> comparison_sets = {
            suite_results,
            quick_results,
            stress_results
        };
        
        std::vector<std::string> labels = {"Full", "Quick", "Stress"};
        
        BenchmarkSuite::generateComparisonReport(comparison_sets, labels, "comparison_report.html");
        std::cout << "✅ Comparison report generated" << std::endl;
        
        std::cout << "\n🎉 All Performance Benchmarking tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
