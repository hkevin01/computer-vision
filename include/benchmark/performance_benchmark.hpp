#pragma once

#include <chrono>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>

namespace stereo_vision::benchmark {

/**
 * @brief Comprehensive performance benchmarking system for stereo vision components
 * 
 * This system provides detailed performance analysis including:
 * - CPU and GPU performance metrics
 * - Memory usage tracking
 * - Frame rate analysis
 * - Algorithm comparison
 * - Hardware optimization suggestions
 */
class PerformanceBenchmark {
public:
    struct SystemInfo {
        std::string cpu_model;
        std::string gpu_model;
        size_t total_memory_mb;
        size_t available_memory_mb;
        std::string opencv_version;
        std::string build_info;
        bool cuda_available;
        bool hip_available;
        int cpu_cores;
        double cpu_frequency_ghz;
    };

    struct BenchmarkConfig {
        int num_iterations = 100;
        int warmup_iterations = 10;
        bool measure_memory = true;
        bool measure_gpu_utilization = true;
        bool save_results = true;
        std::string output_file = "benchmark_results.json";
        cv::Size test_resolution = cv::Size(1920, 1080);
        std::vector<cv::Size> test_resolutions = {
            cv::Size(640, 480),
            cv::Size(1280, 720),
            cv::Size(1920, 1080),
            cv::Size(2560, 1440)
        };
    };

    struct PerformanceMetrics {
        // Timing metrics
        double avg_time_ms = 0.0;
        double min_time_ms = 0.0;
        double max_time_ms = 0.0;
        double std_dev_ms = 0.0;
        double fps = 0.0;
        
        // Memory metrics
        size_t peak_memory_mb = 0;
        size_t avg_memory_mb = 0;
        
        // GPU metrics
        double gpu_utilization_percent = 0.0;
        size_t gpu_memory_used_mb = 0;
        
        // Quality metrics
        double accuracy_score = 0.0;  // Algorithm-specific accuracy
        double robustness_score = 0.0;  // Performance consistency
        
        // System load
        double cpu_usage_percent = 0.0;
        double system_load = 0.0;
        
        // Additional metrics
        std::map<std::string, double> custom_metrics;
    };

    struct BenchmarkResult {
        std::string test_name;
        std::string algorithm_name;
        SystemInfo system_info;
        BenchmarkConfig config;
        std::map<cv::Size, PerformanceMetrics> resolution_metrics;
        double overall_score = 0.0;  // Composite performance score
        std::string timestamp;
        std::vector<std::string> notes;
    };

public:
    explicit PerformanceBenchmark(const BenchmarkConfig& config = BenchmarkConfig{});
    ~PerformanceBenchmark();

    /**
     * @brief Benchmark stereo matching algorithms
     */
    BenchmarkResult benchmarkStereoMatching(
        const std::function<cv::Mat(const cv::Mat&, const cv::Mat&)>& algorithm,
        const std::string& algorithm_name,
        const std::vector<std::pair<cv::Mat, cv::Mat>>& test_images);

    /**
     * @brief Benchmark camera calibration
     */
    BenchmarkResult benchmarkCalibration(
        const std::function<bool(const std::vector<cv::Mat>&)>& calibration_func,
        const std::string& algorithm_name,
        const std::vector<std::vector<cv::Mat>>& calibration_sets);

    /**
     * @brief Benchmark point cloud generation
     */
    BenchmarkResult benchmarkPointCloudGeneration(
        const std::function<void(const cv::Mat&, const cv::Mat&)>& point_cloud_func,
        const std::string& algorithm_name,
        const std::vector<std::pair<cv::Mat, cv::Mat>>& test_data);

    /**
     * @brief Benchmark complete stereo vision pipeline
     */
    BenchmarkResult benchmarkFullPipeline(
        const std::function<void(const cv::Mat&, const cv::Mat&)>& pipeline_func,
        const std::string& pipeline_name,
        const std::vector<std::pair<cv::Mat, cv::Mat>>& test_images);

    /**
     * @brief Compare multiple algorithms
     */
    struct ComparisonResult {
        std::vector<BenchmarkResult> results;
        std::map<std::string, std::string> recommendations;
        std::string best_for_speed;
        std::string best_for_quality;
        std::string best_overall;
    };

    ComparisonResult compareAlgorithms(
        const std::map<std::string, std::function<cv::Mat(const cv::Mat&, const cv::Mat&)>>& algorithms,
        const std::vector<std::pair<cv::Mat, cv::Mat>>& test_images);

    /**
     * @brief Hardware optimization analysis
     */
    struct OptimizationSuggestions {
        std::vector<std::string> cpu_optimizations;
        std::vector<std::string> memory_optimizations;
        std::vector<std::string> gpu_optimizations;
        std::vector<std::string> algorithm_optimizations;
        double potential_speedup_factor = 1.0;
    };

    OptimizationSuggestions analyzeOptimizationOpportunities(const BenchmarkResult& result);

    /**
     * @brief Generate test data for benchmarking
     */
    static std::vector<std::pair<cv::Mat, cv::Mat>> generateTestData(
        const std::vector<cv::Size>& resolutions,
        int samples_per_resolution = 5);

    /**
     * @brief Load real test images from directory
     */
    static std::vector<std::pair<cv::Mat, cv::Mat>> loadTestImages(
        const std::string& directory_path);

    /**
     * @brief Save benchmark results to JSON
     */
    bool saveResults(const BenchmarkResult& result, const std::string& filename) const;
    bool saveComparison(const ComparisonResult& comparison, const std::string& filename) const;

    /**
     * @brief Load benchmark results from JSON
     */
    static BenchmarkResult loadResults(const std::string& filename);
    static ComparisonResult loadComparison(const std::string& filename);

    /**
     * @brief Generate HTML report
     */
    bool generateHTMLReport(const ComparisonResult& comparison, 
                          const std::string& output_file = "benchmark_report.html") const;

    /**
     * @brief Real-time performance monitoring
     */
    class RealtimeMonitor {
    public:
        struct RealtimeMetrics {
            double current_fps = 0.0;
            double avg_fps = 0.0;
            size_t current_memory_mb = 0;
            double cpu_usage = 0.0;
            double gpu_usage = 0.0;
            std::chrono::steady_clock::time_point last_update;
        };

        void startMonitoring();
        void stopMonitoring();
        void recordFrame();
        RealtimeMetrics getCurrentMetrics() const;
        void setUpdateCallback(std::function<void(const RealtimeMetrics&)> callback);

    private:
        std::atomic<bool> monitoring_{false};
        mutable std::mutex metrics_mutex_;
        RealtimeMetrics current_metrics_;
        std::vector<double> recent_frame_times_;
        std::function<void(const RealtimeMetrics&)> update_callback_;
        std::thread monitoring_thread_;
        
        void monitoringLoop();
    };

    /**
     * @brief Automated regression testing
     */
    class RegressionTester {
    public:
        struct RegressionConfig {
            double performance_tolerance = 0.05;  // 5% tolerance
            std::string baseline_file;
            bool auto_update_baseline = false;
        };

        explicit RegressionTester(const RegressionConfig& config);
        
        bool runRegressionTest(const BenchmarkResult& current_result);
        struct RegressionReport {
            bool passed = false;
            std::vector<std::string> performance_regressions;
            std::vector<std::string> improvements;
            double performance_change_percent = 0.0;
        };
        
        RegressionReport getLastReport() const { return last_report_; }

    private:
        RegressionConfig config_;
        BenchmarkResult baseline_result_;
        RegressionReport last_report_;
    };

    /**
     * @brief Get system information
     */
    static SystemInfo getSystemInfo();

    /**
     * @brief Calibrate benchmark timing (account for system overhead)
     */
    void calibrateTiming();

private:
    BenchmarkConfig config_;
    SystemInfo system_info_;
    
    // Timing calibration
    double timing_overhead_ns_ = 0.0;
    
    // Performance monitoring
    std::unique_ptr<RealtimeMonitor> realtime_monitor_;
    
    // Helper methods
    PerformanceMetrics measureFunction(
        const std::function<void()>& func,
        int iterations,
        const std::string& test_name) const;
    
    size_t getCurrentMemoryUsage() const;
    double getCurrentCPUUsage() const;
    double getCurrentGPUUsage() const;
    size_t getGPUMemoryUsage() const;
    
    double calculateCompositeScore(const PerformanceMetrics& metrics) const;
    void addTimestamp(BenchmarkResult& result) const;
    
    // Platform-specific implementations
    class PlatformMonitor;
    std::unique_ptr<PlatformMonitor> platform_monitor_;
};

/**
 * @brief Specialized benchmark for neural network stereo matchers
 */
class NeuralNetworkBenchmark {
public:
    struct ModelMetrics {
        std::string model_name;
        std::string backend;
        double inference_time_ms = 0.0;
        double preprocessing_time_ms = 0.0;
        double postprocessing_time_ms = 0.0;
        size_t model_size_mb = 0;
        double accuracy_score = 0.0;
        bool supports_fp16 = false;
        bool supports_int8 = false;
    };

    /**
     * @brief Benchmark neural network models
     */
    std::vector<ModelMetrics> benchmarkModels(
        const std::vector<std::string>& model_paths,
        const std::vector<std::pair<cv::Mat, cv::Mat>>& test_data);

    /**
     * @brief Find optimal model configuration
     */
    struct OptimalConfig {
        std::string model_path;
        std::string backend;
        bool use_fp16;
        cv::Size input_resolution;
        double expected_fps;
        double expected_accuracy;
    };

    OptimalConfig findOptimalConfiguration(
        double target_fps,
        double min_accuracy,
        const std::vector<std::pair<cv::Mat, cv::Mat>>& validation_data);
};

/**
 * @brief Multi-threaded benchmark execution
 */
class ParallelBenchmark {
public:
    /**
     * @brief Run benchmarks in parallel across multiple threads
     */
    static std::vector<PerformanceBenchmark::BenchmarkResult> runParallel(
        const std::vector<std::function<PerformanceBenchmark::BenchmarkResult()>>& benchmark_tasks,
        int max_threads = std::thread::hardware_concurrency());

    /**
     * @brief Benchmark scalability across different thread counts
     */
    struct ScalabilityResult {
        std::map<int, double> thread_count_to_fps;
        int optimal_thread_count;
        double max_efficiency;
        double scaling_factor;
    };

    static ScalabilityResult benchmarkScalability(
        const std::function<void()>& workload,
        const std::vector<int>& thread_counts = {1, 2, 4, 8, 16});
};

} // namespace stereo_vision::benchmark
