#include "benchmark/performance_benchmark.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <filesystem>

namespace stereovision {
namespace benchmark {

// ============================================================================
// PerformanceBenchmark Implementation
// ============================================================================

PerformanceBenchmark::PerformanceBenchmark() {}

PerformanceBenchmark::~PerformanceBenchmark() {}

void PerformanceBenchmark::addAlgorithm(
    const std::string& name,
    std::function<cv::Mat(const cv::Mat&, const cv::Mat&)> stereo_func) {
    
    algorithms_[name] = stereo_func;
}

BenchmarkResult PerformanceBenchmark::runBenchmark(
    const std::string& algorithm_name,
    const std::vector<std::pair<cv::Mat, cv::Mat>>& test_images,
    int iterations) {
    
    auto it = algorithms_.find(algorithm_name);
    if (it == algorithms_.end()) {
        throw std::runtime_error("Algorithm not found: " + algorithm_name);
    }
    
    BenchmarkResult result;
    result.algorithm_name = algorithm_name;
    result.total_iterations = iterations * test_images.size();
    
    std::vector<double> processing_times;
    std::vector<MemoryUsage> memory_snapshots;
    
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < test_images.size(); ++i) {
            const auto& image_pair = test_images[i];
            
            // Memory usage before processing
            MemoryUsage memory_before = getCurrentMemoryUsage();
            
            // Time the processing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            cv::Mat disparity = it->second(image_pair.first, image_pair.second);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            
            // Memory usage after processing
            MemoryUsage memory_after = getCurrentMemoryUsage();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);
            processing_times.push_back(duration.count() / 1000.0); // Convert to ms
            
            // Store memory delta
            MemoryUsage memory_delta;
            memory_delta.peak_memory_mb = memory_after.peak_memory_mb - memory_before.peak_memory_mb;
            memory_delta.current_memory_mb = memory_after.current_memory_mb - memory_before.current_memory_mb;
            memory_snapshots.push_back(memory_delta);
        }
    }
    
    // Calculate statistics
    std::sort(processing_times.begin(), processing_times.end());
    
    result.min_time_ms = processing_times.front();
    result.max_time_ms = processing_times.back();
    result.avg_time_ms = std::accumulate(processing_times.begin(), 
                                        processing_times.end(), 0.0) / processing_times.size();
    
    // Calculate median
    size_t median_idx = processing_times.size() / 2;
    if (processing_times.size() % 2 == 0) {
        result.median_time_ms = (processing_times[median_idx - 1] + processing_times[median_idx]) / 2.0;
    } else {
        result.median_time_ms = processing_times[median_idx];
    }
    
    // Calculate 95th percentile
    size_t p95_idx = static_cast<size_t>(processing_times.size() * 0.95);
    result.p95_time_ms = processing_times[std::min(p95_idx, processing_times.size() - 1)];
    
    // Calculate fps metrics
    result.avg_fps = 1000.0 / result.avg_time_ms;
    result.min_fps = 1000.0 / result.max_time_ms;
    result.max_fps = 1000.0 / result.min_time_ms;
    
    // Memory statistics
    double total_peak_memory = 0, total_current_memory = 0;
    for (const auto& mem : memory_snapshots) {
        total_peak_memory += mem.peak_memory_mb;
        total_current_memory += mem.current_memory_mb;
    }
    
    result.memory_usage.peak_memory_mb = total_peak_memory / memory_snapshots.size();
    result.memory_usage.current_memory_mb = total_current_memory / memory_snapshots.size();
    
    return result;
}

ComparisonReport PerformanceBenchmark::compareAlgorithms(
    const std::vector<std::pair<cv::Mat, cv::Mat>>& test_images,
    int iterations) {
    
    ComparisonReport report;
    report.test_image_count = test_images.size();
    report.iterations_per_image = iterations;
    
    // Run benchmarks for all algorithms
    for (const auto& algo_pair : algorithms_) {
        BenchmarkResult result = runBenchmark(algo_pair.first, test_images, iterations);
        report.results.push_back(result);
    }
    
    // Sort by average performance
    std::sort(report.results.begin(), report.results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.avg_time_ms < b.avg_time_ms;
        });
    
    // Find best performing algorithm
    if (!report.results.empty()) {
        report.best_algorithm = report.results[0].algorithm_name;
    }
    
    return report;
}

std::string PerformanceBenchmark::generateReport(const ComparisonReport& report) {
    std::stringstream ss;
    
    ss << "# Stereo Vision Performance Benchmark Report\n\n";
    ss << "## Test Configuration\n";
    ss << "- Test Images: " << report.test_image_count << "\n";
    ss << "- Iterations per Image: " << report.iterations_per_image << "\n";
    ss << "- Total Tests: " << (report.test_image_count * report.iterations_per_image) << "\n";
    ss << "- Best Algorithm: " << report.best_algorithm << "\n\n";
    
    ss << "## Performance Results\n\n";
    ss << "| Algorithm | Avg Time (ms) | Min/Max (ms) | Median (ms) | 95th %ile | Avg FPS | Memory (MB) |\n";
    ss << "|-----------|---------------|--------------|-------------|-----------|---------|-------------|\n";
    
    for (const auto& result : report.results) {
        ss << "| " << result.algorithm_name
           << " | " << std::fixed << std::setprecision(2) << result.avg_time_ms
           << " | " << result.min_time_ms << "/" << result.max_time_ms
           << " | " << result.median_time_ms
           << " | " << result.p95_time_ms
           << " | " << std::setprecision(1) << result.avg_fps
           << " | " << std::setprecision(1) << result.memory_usage.peak_memory_mb
           << " |\n";
    }
    
    ss << "\n## Detailed Analysis\n\n";
    
    for (const auto& result : report.results) {
        ss << "### " << result.algorithm_name << "\n";
        ss << "- **Processing Time**: " << result.avg_time_ms << " ms average\n";
        ss << "- **Frame Rate**: " << result.avg_fps << " FPS average\n";
        ss << "- **Performance Range**: " << result.min_fps << " - " << result.max_fps << " FPS\n";
        ss << "- **Memory Usage**: " << result.memory_usage.peak_memory_mb << " MB peak\n";
        ss << "- **Consistency**: " << (result.max_time_ms - result.min_time_ms) << " ms variance\n\n";
    }
    
    return ss.str();
}

std::string PerformanceBenchmark::generateHTMLReport(const ComparisonReport& report) {
    std::stringstream ss;
    
    ss << R"(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stereo Vision Performance Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; text-align: center; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .summary { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .best-algorithm { background-color: #2ecc71 !important; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Stereo Vision Performance Benchmark Report</h1>
        
        <div class="summary">
            <h2>📊 Test Summary</h2>
            <p><strong>Test Images:</strong> )" << report.test_image_count << R"(</p>
            <p><strong>Iterations per Image:</strong> )" << report.iterations_per_image << R"(</p>
            <p><strong>Total Tests:</strong> )" << (report.test_image_count * report.iterations_per_image) << R"(</p>
            <p><strong>🏆 Best Algorithm:</strong> )" << report.best_algorithm << R"(</p>
        </div>

        <h2>⚡ Performance Comparison</h2>
        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>

        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Algorithm</th>
                    <th>Avg Time (ms)</th>
                    <th>Min Time (ms)</th>
                    <th>Max Time (ms)</th>
                    <th>Median (ms)</th>
                    <th>95th %ile (ms)</th>
                    <th>Avg FPS</th>
                    <th>Memory (MB)</th>
                </tr>
            </thead>
            <tbody>)";
    
    for (size_t i = 0; i < report.results.size(); ++i) {
        const auto& result = report.results[i];
        std::string row_class = (i == 0) ? " class=\"best-algorithm\"" : "";
        
        ss << "                <tr" << row_class << ">\n";
        ss << "                    <td>" << result.algorithm_name << "</td>\n";
        ss << "                    <td>" << std::fixed << std::setprecision(2) << result.avg_time_ms << "</td>\n";
        ss << "                    <td>" << result.min_time_ms << "</td>\n";
        ss << "                    <td>" << result.max_time_ms << "</td>\n";
        ss << "                    <td>" << result.median_time_ms << "</td>\n";
        ss << "                    <td>" << result.p95_time_ms << "</td>\n";
        ss << "                    <td>" << std::setprecision(1) << result.avg_fps << "</td>\n";
        ss << "                    <td>" << result.memory_usage.peak_memory_mb << "</td>\n";
        ss << "                </tr>\n";
    }
    
    ss << R"(            </tbody>
        </table>

        <h2>📈 FPS Comparison</h2>
        <div class="chart-container">
            <canvas id="fpsChart"></canvas>
        </div>

        <h2>💾 Memory Usage</h2>
        <div class="chart-container">
            <canvas id="memoryChart"></canvas>
        </div>
    </div>

    <script>
        // Performance Chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(perfCtx, {
            type: 'bar',
            data: {
                labels: [)";
    
    // Add algorithm names for charts
    for (size_t i = 0; i < report.results.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "'" << report.results[i].algorithm_name << "'";
    }
    
    ss << R"(],
                datasets: [{
                    label: 'Average Processing Time (ms)',
                    data: [)";
    
    // Add processing times
    for (size_t i = 0; i < report.results.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << std::fixed << std::setprecision(2) << report.results[i].avg_time_ms;
    }
    
    ss << R"(],
                    backgroundColor: '#3498db',
                    borderColor: '#2980b9',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // FPS Chart
        const fpsCtx = document.getElementById('fpsChart').getContext('2d');
        new Chart(fpsCtx, {
            type: 'line',
            data: {
                labels: [)";
    
    // Add algorithm names again
    for (size_t i = 0; i < report.results.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "'" << report.results[i].algorithm_name << "'";
    }
    
    ss << R"(],
                datasets: [{
                    label: 'Average FPS',
                    data: [)";
    
    // Add FPS data
    for (size_t i = 0; i < report.results.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << std::fixed << std::setprecision(1) << report.results[i].avg_fps;
    }
    
    ss << R"(],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // Memory Chart
        const memCtx = document.getElementById('memoryChart').getContext('2d');
        new Chart(memCtx, {
            type: 'doughnut',
            data: {
                labels: [)";
    
    // Add algorithm names for memory chart
    for (size_t i = 0; i < report.results.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "'" << report.results[i].algorithm_name << "'";
    }
    
    ss << R"(],
                datasets: [{
                    data: [)";
    
    // Add memory data
    for (size_t i = 0; i < report.results.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << std::fixed << std::setprecision(1) << report.results[i].memory_usage.peak_memory_mb;
    }
    
    ss << R"(],
                    backgroundColor: [
                        '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    </script>
</body>
</html>)";
    
    return ss.str();
}

MemoryUsage PerformanceBenchmark::getCurrentMemoryUsage() {
    MemoryUsage usage;
    
    // Platform-specific memory usage detection
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            int kb;
            iss >> kb;
            usage.current_memory_mb = kb / 1024.0;
        } else if (line.substr(0, 6) == "VmPeak:") {
            std::istringstream iss(line.substr(7));
            int kb;
            iss >> kb;
            usage.peak_memory_mb = kb / 1024.0;
        }
    }
#else
    // Fallback for other platforms
    usage.current_memory_mb = 0;
    usage.peak_memory_mb = 0;
#endif
    
    return usage;
}

// ============================================================================
// RealtimeMonitor Implementation
// ============================================================================

RealtimeMonitor::RealtimeMonitor() 
    : is_monitoring_(false)
    , sample_interval_ms_(100) {}

RealtimeMonitor::~RealtimeMonitor() {
    stopMonitoring();
}

void RealtimeMonitor::startMonitoring(
    std::function<cv::Mat(const cv::Mat&, const cv::Mat&)> algorithm) {
    
    if (is_monitoring_) {
        return;
    }
    
    algorithm_ = algorithm;
    is_monitoring_ = true;
    
    monitoring_thread_ = std::thread(&RealtimeMonitor::monitoringLoop, this);
}

void RealtimeMonitor::stopMonitoring() {
    is_monitoring_ = false;
    
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
}

void RealtimeMonitor::addFramePair(const cv::Mat& left, const cv::Mat& right) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // Keep only the latest frame pair to avoid memory buildup
    latest_left_ = left.clone();
    latest_right_ = right.clone();
    has_new_frame_ = true;
}

RealtimeStats RealtimeMonitor::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return current_stats_;
}

void RealtimeMonitor::setSampleInterval(int interval_ms) {
    sample_interval_ms_ = interval_ms;
}

void RealtimeMonitor::monitoringLoop() {
    std::vector<double> processing_times;
    std::vector<double> memory_usage;
    
    while (is_monitoring_) {
        cv::Mat left, right;
        bool has_frame = false;
        
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            if (has_new_frame_) {
                left = latest_left_.clone();
                right = latest_right_.clone();
                has_frame = true;
                has_new_frame_ = false;
            }
        }
        
        if (has_frame && algorithm_) {
            auto start_time = std::chrono::high_resolution_clock::now();
            MemoryUsage mem_before = getCurrentMemoryUsage();
            
            cv::Mat result = algorithm_(left, right);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            MemoryUsage mem_after = getCurrentMemoryUsage();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);
            double processing_time_ms = duration.count() / 1000.0;
            
            processing_times.push_back(processing_time_ms);
            memory_usage.push_back(mem_after.current_memory_mb);
            
            // Keep only recent samples (last 100 samples)
            if (processing_times.size() > 100) {
                processing_times.erase(processing_times.begin());
                memory_usage.erase(memory_usage.begin());
            }
            
            // Update statistics
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                updateStats(processing_times, memory_usage);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms_));
    }
}

void RealtimeMonitor::updateStats(
    const std::vector<double>& processing_times,
    const std::vector<double>& memory_usage) {
    
    if (processing_times.empty()) {
        return;
    }
    
    // Calculate processing time statistics
    double total_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0);
    current_stats_.avg_processing_time_ms = total_time / processing_times.size();
    
    auto minmax_time = std::minmax_element(processing_times.begin(), processing_times.end());
    current_stats_.min_processing_time_ms = *minmax_time.first;
    current_stats_.max_processing_time_ms = *minmax_time.second;
    
    current_stats_.current_fps = 1000.0 / current_stats_.avg_processing_time_ms;
    
    // Calculate memory statistics
    if (!memory_usage.empty()) {
        double total_memory = std::accumulate(memory_usage.begin(), memory_usage.end(), 0.0);
        current_stats_.avg_memory_usage_mb = total_memory / memory_usage.size();
        
        auto minmax_memory = std::minmax_element(memory_usage.begin(), memory_usage.end());
        current_stats_.min_memory_usage_mb = *minmax_memory.first;
        current_stats_.max_memory_usage_mb = *minmax_memory.second;
    }
    
    current_stats_.sample_count = processing_times.size();
}

MemoryUsage RealtimeMonitor::getCurrentMemoryUsage() {
    // Reuse the implementation from PerformanceBenchmark
    MemoryUsage usage;
    
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            int kb;
            iss >> kb;
            usage.current_memory_mb = kb / 1024.0;
        } else if (line.substr(0, 6) == "VmPeak:") {
            std::istringstream iss(line.substr(7));
            int kb;
            iss >> kb;
            usage.peak_memory_mb = kb / 1024.0;
        }
    }
#else
    usage.current_memory_mb = 0;
    usage.peak_memory_mb = 0;
#endif
    
    return usage;
}

// ============================================================================
// RegressionTester Implementation
// ============================================================================

RegressionTester::RegressionTester() {}

RegressionTester::~RegressionTester() {}

void RegressionTester::addBaselineResult(const std::string& algorithm_name, 
                                       const BenchmarkResult& baseline) {
    baselines_[algorithm_name] = baseline;
}

bool RegressionTester::loadBaselinesFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Simple CSV format: algorithm,avg_time,avg_fps,memory
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string algorithm, avg_time_str, avg_fps_str, memory_str;
        
        if (std::getline(ss, algorithm, ',') &&
            std::getline(ss, avg_time_str, ',') &&
            std::getline(ss, avg_fps_str, ',') &&
            std::getline(ss, memory_str, ',')) {
            
            BenchmarkResult baseline;
            baseline.algorithm_name = algorithm;
            baseline.avg_time_ms = std::stod(avg_time_str);
            baseline.avg_fps = std::stod(avg_fps_str);
            baseline.memory_usage.peak_memory_mb = std::stod(memory_str);
            
            baselines_[algorithm] = baseline;
        }
    }
    
    return !baselines_.empty();
}

bool RegressionTester::saveBaselinesToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "algorithm,avg_time_ms,avg_fps,memory_mb\n";
    
    for (const auto& pair : baselines_) {
        const auto& result = pair.second;
        file << result.algorithm_name << ","
             << std::fixed << std::setprecision(3) << result.avg_time_ms << ","
             << result.avg_fps << ","
             << result.memory_usage.peak_memory_mb << "\n";
    }
    
    return true;
}

RegressionReport RegressionTester::checkRegression(
    const std::vector<BenchmarkResult>& current_results,
    double performance_threshold,
    double memory_threshold) {
    
    RegressionReport report;
    report.performance_threshold = performance_threshold;
    report.memory_threshold = memory_threshold;
    
    for (const auto& current : current_results) {
        auto baseline_it = baselines_.find(current.algorithm_name);
        if (baseline_it == baselines_.end()) {
            continue; // No baseline for this algorithm
        }
        
        const auto& baseline = baseline_it->second;
        
        RegressionResult reg_result;
        reg_result.algorithm_name = current.algorithm_name;
        reg_result.baseline_time_ms = baseline.avg_time_ms;
        reg_result.current_time_ms = current.avg_time_ms;
        reg_result.baseline_memory_mb = baseline.memory_usage.peak_memory_mb;
        reg_result.current_memory_mb = current.memory_usage.peak_memory_mb;
        
        // Calculate performance change
        reg_result.performance_change_percent = 
            ((current.avg_time_ms - baseline.avg_time_ms) / baseline.avg_time_ms) * 100.0;
        
        // Calculate memory change
        if (baseline.memory_usage.peak_memory_mb > 0) {
            reg_result.memory_change_percent = 
                ((current.memory_usage.peak_memory_mb - baseline.memory_usage.peak_memory_mb) 
                 / baseline.memory_usage.peak_memory_mb) * 100.0;
        } else {
            reg_result.memory_change_percent = 0.0;
        }
        
        // Check for regressions
        reg_result.performance_regression = 
            reg_result.performance_change_percent > performance_threshold;
        reg_result.memory_regression = 
            reg_result.memory_change_percent > memory_threshold;
        
        report.results.push_back(reg_result);
        
        if (reg_result.performance_regression || reg_result.memory_regression) {
            report.has_regressions = true;
        }
    }
    
    return report;
}

std::string RegressionTester::generateRegressionReport(const RegressionReport& report) {
    std::stringstream ss;
    
    ss << "# Regression Test Report\n\n";
    ss << "## Test Configuration\n";
    ss << "- Performance Threshold: " << report.performance_threshold << "%\n";
    ss << "- Memory Threshold: " << report.memory_threshold << "%\n";
    ss << "- **Regressions Detected: " << (report.has_regressions ? "YES" : "NO") << "**\n\n";
    
    if (report.has_regressions) {
        ss << "## ⚠️ Regressions Found\n\n";
        
        for (const auto& result : report.results) {
            if (result.performance_regression || result.memory_regression) {
                ss << "### " << result.algorithm_name << "\n";
                
                if (result.performance_regression) {
                    ss << "- **Performance Regression**: " 
                       << std::fixed << std::setprecision(1) 
                       << result.performance_change_percent << "% slower\n";
                    ss << "  - Baseline: " << result.baseline_time_ms << " ms\n";
                    ss << "  - Current: " << result.current_time_ms << " ms\n";
                }
                
                if (result.memory_regression) {
                    ss << "- **Memory Regression**: " 
                       << std::fixed << std::setprecision(1) 
                       << result.memory_change_percent << "% more memory\n";
                    ss << "  - Baseline: " << result.baseline_memory_mb << " MB\n";
                    ss << "  - Current: " << result.current_memory_mb << " MB\n";
                }
                
                ss << "\n";
            }
        }
    }
    
    ss << "## All Results\n\n";
    ss << "| Algorithm | Performance Change | Memory Change | Status |\n";
    ss << "|-----------|-------------------|---------------|--------|\n";
    
    for (const auto& result : report.results) {
        std::string status = "✅ PASS";
        if (result.performance_regression || result.memory_regression) {
            status = "❌ FAIL";
        }
        
        ss << "| " << result.algorithm_name
           << " | " << std::fixed << std::setprecision(1) << result.performance_change_percent << "%"
           << " | " << result.memory_change_percent << "%"
           << " | " << status << " |\n";
    }
    
    return ss.str();
}

} // namespace benchmark
} // namespace stereovision
