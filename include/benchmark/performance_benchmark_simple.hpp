#pragma once

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace stereovision {
namespace benchmark {

struct BenchmarkResult {
    std::string test_name;
    double avg_fps;
    double min_fps;
    double max_fps;
    double avg_latency_ms;
    double memory_usage_mb;
    double cpu_usage_percent;
    bool success;
    std::string error_message;
    
    BenchmarkResult() : avg_fps(0.0), min_fps(0.0), max_fps(0.0), 
                       avg_latency_ms(0.0), memory_usage_mb(0.0), 
                       cpu_usage_percent(0.0), success(false) {}
};

struct SystemInfo {
    std::string os_name;
    std::string cpu_model;
    int cpu_cores;
    std::string gpu_model;
    size_t total_memory_mb;
    std::string opencv_version;
    
    SystemInfo();
};

class PerformanceBenchmark {
public:
    PerformanceBenchmark();
    ~PerformanceBenchmark() = default;

    // Configuration
    void setTestDuration(int seconds) { test_duration_seconds_ = seconds; }
    void setWarmupFrames(int frames) { warmup_frames_ = frames; }
    void setTestImages(const cv::Mat& left, const cv::Mat& right);
    
    // Core benchmarking
    BenchmarkResult benchmarkStereoMatching(const std::string& algorithm_name,
                                           std::function<cv::Mat(const cv::Mat&, const cv::Mat&)> algorithm);
    
    BenchmarkResult benchmarkNeuralMatcher(const std::string& model_name);
    BenchmarkResult benchmarkMultiCamera(int num_cameras = 2);
    
    // Comprehensive testing
    std::vector<BenchmarkResult> runFullBenchmarkSuite();
    std::vector<BenchmarkResult> runStressTest(int duration_minutes = 10);
    
    // Regression testing
    bool runRegressionTest(const std::string& baseline_file);
    void saveBaseline(const std::string& filename, const std::vector<BenchmarkResult>& results);
    
    // Results and reporting
    void generateHTMLReport(const std::vector<BenchmarkResult>& results, const std::string& filename);
    void generateCSVReport(const std::vector<BenchmarkResult>& results, const std::string& filename);
    void printSummary(const std::vector<BenchmarkResult>& results);
    
    // System monitoring
    SystemInfo getSystemInfo() const;
    double getCurrentCPUUsage();
    double getCurrentMemoryUsage();
    double getCurrentGPUUsage();

private:
    int test_duration_seconds_;
    int warmup_frames_;
    cv::Mat test_left_image_;
    cv::Mat test_right_image_;
    SystemInfo system_info_;
    
    // Internal helpers
    BenchmarkResult runTimedTest(const std::string& test_name, 
                                std::function<bool()> test_function);
    void warmupSystem();
    double measureSystemMetric(std::function<double()> metric_function);
    
    // Monitoring helpers
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<double> frame_times_;
    std::vector<double> memory_samples_;
    std::vector<double> cpu_samples_;
};

class RealtimeBenchmark {
public:
    RealtimeBenchmark();
    ~RealtimeBenchmark();
    
    // Real-time monitoring
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const { return is_monitoring_; }
    
    // Metrics collection
    void recordFrame(double processing_time_ms);
    void recordMemoryUsage(double memory_mb);
    void recordCPUUsage(double cpu_percent);
    
    // Live statistics
    double getCurrentFPS() const;
    double getAverageLatency() const;
    double getPeakMemoryUsage() const;
    
    // Alerts and thresholds
    void setPerformanceThresholds(double min_fps, double max_latency_ms, double max_memory_mb);
    bool hasPerformanceIssues() const;
    std::vector<std::string> getPerformanceAlerts() const;

private:
    std::atomic<bool> is_monitoring_;
    std::thread monitoring_thread_;
    
    // Performance data
    std::vector<double> recent_frame_times_;
    std::vector<double> recent_memory_usage_;
    std::vector<double> recent_cpu_usage_;
    mutable std::mutex data_mutex_;
    
    // Thresholds
    double min_fps_threshold_;
    double max_latency_threshold_;
    double max_memory_threshold_;
    
    // Monitoring thread
    void monitoringLoop();
    void collectSystemMetrics();
};

class BenchmarkSuite {
public:
    static std::vector<BenchmarkResult> runQuickBenchmark();
    static std::vector<BenchmarkResult> runFullBenchmark();
    static std::vector<BenchmarkResult> runStressBenchmark(int duration_minutes = 30);
    
    // Specific component benchmarks
    static BenchmarkResult benchmarkStereoAlgorithms();
    static BenchmarkResult benchmarkNeuralNetworks();
    static BenchmarkResult benchmarkMultiCameraSystem();
    static BenchmarkResult benchmarkCalibration();
    
    // Utility functions
    static void generateComparisonReport(const std::vector<std::vector<BenchmarkResult>>& result_sets,
                                       const std::vector<std::string>& labels,
                                       const std::string& output_file);
};

} // namespace benchmark
} // namespace stereovision
