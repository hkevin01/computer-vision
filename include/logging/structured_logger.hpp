#pragma once

#include <string>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <mutex>

#ifdef SPDLOG_ACTIVE_LEVEL
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/json_sink.h>
#include <spdlog/fmt/ostr.h>
#endif

namespace cv_stereo {

struct SessionMetrics {
    std::string session_uuid;
    std::string model_name;
    std::string provider_name;

    // Timing metrics
    double preprocessing_time_ms = 0.0;
    double inference_time_ms = 0.0;
    double postprocessing_time_ms = 0.0;
    double total_time_ms = 0.0;

    // Memory metrics
    size_t input_memory_mb = 0;
    size_t output_memory_mb = 0;
    size_t peak_memory_mb = 0;

    // Quality metrics
    double disparity_range_min = 0.0;
    double disparity_range_max = 0.0;
    double confidence_mean = 0.0;
    size_t valid_pixels = 0;
    size_t total_pixels = 0;

    // System context
    std::string gpu_name;
    std::string driver_version;
    std::string opencv_version;
    std::chrono::system_clock::time_point timestamp;

    // Error tracking
    std::string error_message;
    bool successful = true;
};

struct SystemInfo {
    std::string cpu_model;
    size_t total_memory_gb;
    std::string os_version;
    std::vector<std::string> available_providers;
    std::string build_config;  // Debug/Release
    std::string git_commit;
};

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CRITICAL = 5
};

class StructuredLogger {
public:
    static StructuredLogger& instance();

    // Initialize logging system
    bool initialize(const std::string& log_dir = "logs/",
                   LogLevel console_level = LogLevel::INFO,
                   LogLevel file_level = LogLevel::DEBUG,
                   bool enable_json_sink = true);

    // Session management
    std::string start_session(const std::string& operation_type = "inference");
    void end_session(const std::string& session_uuid);
    void log_session_metrics(const SessionMetrics& metrics);

    // Structured logging with context
    void log_info(const std::string& message,
                  const std::unordered_map<std::string, std::string>& context = {});
    void log_warning(const std::string& message,
                     const std::unordered_map<std::string, std::string>& context = {});
    void log_error(const std::string& message,
                   const std::unordered_map<std::string, std::string>& context = {});

    // Performance logging
    void log_timing(const std::string& operation,
                   double duration_ms,
                   const std::string& session_uuid = "");

    void log_memory_usage(size_t memory_mb,
                         const std::string& component = "",
                         const std::string& session_uuid = "");

    // System diagnostics
    void log_system_info();
    void log_gpu_diagnostics();
    void log_model_load_event(const std::string& model_name,
                             const std::string& model_path,
                             bool success,
                             const std::string& error_msg = "");

    // Configuration
    void set_log_level(LogLevel level);
    void enable_console_logging(bool enable);
    void enable_file_logging(bool enable);
    void enable_json_logging(bool enable);

    // Metrics aggregation
    std::vector<SessionMetrics> get_session_history(
        const std::string& model_name = "",
        const std::chrono::hours& time_window = std::chrono::hours(24)) const;

    // Cleanup
    void flush();
    void shutdown();

private:
    StructuredLogger() = default;
    ~StructuredLogger();

    std::string generate_session_uuid();
    std::string get_timestamp_iso8601();
    SystemInfo collect_system_info();
    std::string format_json_context(const std::unordered_map<std::string, std::string>& context);

#ifdef SPDLOG_ACTIVE_LEVEL
    std::shared_ptr<spdlog::logger> console_logger_;
    std::shared_ptr<spdlog::logger> file_logger_;
    std::shared_ptr<spdlog::logger> json_logger_;
#endif

    bool initialized_ = false;
    std::string log_directory_;
    LogLevel current_level_ = LogLevel::INFO;

    // Session tracking
    std::unordered_map<std::string, std::chrono::system_clock::time_point> active_sessions_;
    std::vector<SessionMetrics> session_history_;

    // System context cache
    SystemInfo system_info_;
    bool system_info_cached_ = false;

    mutable std::mutex logger_mutex_;
};

// Convenience macros for structured logging
#define CV_LOG_INFO(msg, ...) \
    cv_stereo::StructuredLogger::instance().log_info(msg, ##__VA_ARGS__)

#define CV_LOG_WARN(msg, ...) \
    cv_stereo::StructuredLogger::instance().log_warning(msg, ##__VA_ARGS__)

#define CV_LOG_ERROR(msg, ...) \
    cv_stereo::StructuredLogger::instance().log_error(msg, ##__VA_ARGS__)

#define CV_LOG_TIMING(op, duration_ms, session) \
    cv_stereo::StructuredLogger::instance().log_timing(op, duration_ms, session)

#define CV_LOG_MEMORY(memory_mb, component, session) \
    cv_stereo::StructuredLogger::instance().log_memory_usage(memory_mb, component, session)

// RAII timing helper
class ScopedTimer {
public:
    ScopedTimer(const std::string& operation_name, const std::string& session_uuid = "")
        : operation_(operation_name), session_(session_uuid),
          start_time_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        CV_LOG_TIMING(operation_, duration_ms, session_);
    }

private:
    std::string operation_;
    std::string session_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

#define CV_TIMER(name) cv_stereo::ScopedTimer timer_##__LINE__(name)
#define CV_TIMER_SESSION(name, session) cv_stereo::ScopedTimer timer_##__LINE__(name, session)

} // namespace cv_stereo
