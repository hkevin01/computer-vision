#include "logging/structured_logger.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <iomanip>

#ifdef SPDLOG_ACTIVE_LEVEL
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/rotating_file_sink.h>
#endif

#ifdef __linux__
#include <sys/utsname.h>
#include <fstream>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace cv_stereo {

StructuredLogger& StructuredLogger::instance() {
    static StructuredLogger instance;
    return instance;
}

StructuredLogger::~StructuredLogger() {
    shutdown();
}

bool StructuredLogger::initialize(const std::string& log_dir,
                                 LogLevel console_level,
                                 LogLevel file_level,
                                 bool enable_json_sink) {
    std::lock_guard<std::mutex> lock(logger_mutex_);

    if (initialized_) {
        return true;
    }

    log_directory_ = log_dir;
    current_level_ = console_level;

    try {
        // Create log directory
        fs::create_directories(log_dir);

#ifdef SPDLOG_ACTIVE_LEVEL
        // Console logger with colored output
        console_logger_ = spdlog::stdout_color_mt("console");
        console_logger_->set_level(static_cast<spdlog::level::level_enum>(console_level));
        console_logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");

        // File logger with rotation
        std::string log_file = log_dir + "stereo_vision.log";
        file_logger_ = spdlog::rotating_logger_mt("file", log_file, 1024 * 1024 * 10, 5); // 10MB, 5 files
        file_logger_->set_level(static_cast<spdlog::level::level_enum>(file_level));
        file_logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%s:%#] %v");

        // JSON logger for structured data
        if (enable_json_sink) {
            std::string json_file = log_dir + "metrics.jsonl";
            auto json_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(json_file, true);
            json_logger_ = std::make_shared<spdlog::logger>("json", json_sink);
            json_logger_->set_level(spdlog::level::trace);
            json_logger_->set_pattern("%v"); // Raw JSON output
            spdlog::register_logger(json_logger_);
        }

        // Set default logger
        spdlog::set_default_logger(console_logger_);
        spdlog::flush_every(std::chrono::seconds(5));
#endif

        // Collect system information
        system_info_ = collect_system_info();
        system_info_cached_ = true;

        initialized_ = true;

        // Log initialization
        log_info("StructuredLogger initialized", {
            {"log_directory", log_dir},
            {"console_level", std::to_string(static_cast<int>(console_level))},
            {"file_level", std::to_string(static_cast<int>(file_level))},
            {"json_enabled", enable_json_sink ? "true" : "false"}
        });

        log_system_info();

        return true;

    } catch (const std::exception& e) {
        // Fallback to stderr if logging setup fails
        std::cerr << "Failed to initialize StructuredLogger: " << e.what() << std::endl;
        return false;
    }
}

std::string StructuredLogger::start_session(const std::string& operation_type) {
    std::lock_guard<std::mutex> lock(logger_mutex_);

    std::string session_uuid = generate_session_uuid();
    active_sessions_[session_uuid] = std::chrono::system_clock::now();

    log_info("Session started", {
        {"session_uuid", session_uuid},
        {"operation_type", operation_type},
        {"timestamp", get_timestamp_iso8601()}
    });

    return session_uuid;
}

void StructuredLogger::end_session(const std::string& session_uuid) {
    std::lock_guard<std::mutex> lock(logger_mutex_);

    auto it = active_sessions_.find(session_uuid);
    if (it != active_sessions_.end()) {
        auto start_time = it->second;
        auto end_time = std::chrono::system_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        log_info("Session ended", {
            {"session_uuid", session_uuid},
            {"duration_ms", std::to_string(duration_ms)},
            {"timestamp", get_timestamp_iso8601()}
        });

        active_sessions_.erase(it);
    }
}

void StructuredLogger::log_session_metrics(const SessionMetrics& metrics) {
    std::lock_guard<std::mutex> lock(logger_mutex_);

    // Add to history
    session_history_.push_back(metrics);

    // Keep only last 1000 sessions
    if (session_history_.size() > 1000) {
        session_history_.erase(session_history_.begin());
    }

#ifdef SPDLOG_ACTIVE_LEVEL
    if (json_logger_) {
        // Create structured JSON log entry
        std::ostringstream json_entry;
        json_entry << "{"
                   << "\"type\":\"session_metrics\","
                   << "\"session_uuid\":\"" << metrics.session_uuid << "\","
                   << "\"model_name\":\"" << metrics.model_name << "\","
                   << "\"provider_name\":\"" << metrics.provider_name << "\","
                   << "\"timing\":{"
                   << "\"preprocessing_ms\":" << metrics.preprocessing_time_ms << ","
                   << "\"inference_ms\":" << metrics.inference_time_ms << ","
                   << "\"postprocessing_ms\":" << metrics.postprocessing_time_ms << ","
                   << "\"total_ms\":" << metrics.total_time_ms
                   << "},"
                   << "\"memory\":{"
                   << "\"input_mb\":" << metrics.input_memory_mb << ","
                   << "\"output_mb\":" << metrics.output_memory_mb << ","
                   << "\"peak_mb\":" << metrics.peak_memory_mb
                   << "},"
                   << "\"quality\":{"
                   << "\"disparity_min\":" << metrics.disparity_range_min << ","
                   << "\"disparity_max\":" << metrics.disparity_range_max << ","
                   << "\"confidence_mean\":" << metrics.confidence_mean << ","
                   << "\"valid_pixels\":" << metrics.valid_pixels << ","
                   << "\"total_pixels\":" << metrics.total_pixels
                   << "},"
                   << "\"system\":{"
                   << "\"gpu_name\":\"" << metrics.gpu_name << "\","
                   << "\"driver_version\":\"" << metrics.driver_version << "\","
                   << "\"opencv_version\":\"" << metrics.opencv_version << "\""
                   << "},"
                   << "\"successful\":" << (metrics.successful ? "true" : "false") << ","
                   << "\"error_message\":\"" << metrics.error_message << "\","
                   << "\"timestamp\":\"" << get_timestamp_iso8601() << "\""
                   << "}";

        json_logger_->info(json_entry.str());
    }
#endif

    // Also log summary to console/file
    if (metrics.successful) {
        log_info("Session metrics recorded", {
            {"session_uuid", metrics.session_uuid},
            {"model", metrics.model_name},
            {"provider", metrics.provider_name},
            {"total_time_ms", std::to_string(metrics.total_time_ms)},
            {"inference_time_ms", std::to_string(metrics.inference_time_ms)}
        });
    } else {
        log_error("Session failed", {
            {"session_uuid", metrics.session_uuid},
            {"model", metrics.model_name},
            {"provider", metrics.provider_name},
            {"error", metrics.error_message}
        });
    }
}

void StructuredLogger::log_info(const std::string& message,
                               const std::unordered_map<std::string, std::string>& context) {
    if (!initialized_) return;

#ifdef SPDLOG_ACTIVE_LEVEL
    if (context.empty()) {
        console_logger_->info(message);
        file_logger_->info(message);
    } else {
        std::string formatted_msg = message + " " + format_json_context(context);
        console_logger_->info(formatted_msg);
        file_logger_->info(formatted_msg);
    }
#else
    std::cout << "[INFO] " << message;
    if (!context.empty()) {
        std::cout << " " << format_json_context(context);
    }
    std::cout << std::endl;
#endif
}

void StructuredLogger::log_warning(const std::string& message,
                                  const std::unordered_map<std::string, std::string>& context) {
    if (!initialized_) return;

#ifdef SPDLOG_ACTIVE_LEVEL
    if (context.empty()) {
        console_logger_->warn(message);
        file_logger_->warn(message);
    } else {
        std::string formatted_msg = message + " " + format_json_context(context);
        console_logger_->warn(formatted_msg);
        file_logger_->warn(formatted_msg);
    }
#else
    std::cout << "[WARN] " << message;
    if (!context.empty()) {
        std::cout << " " << format_json_context(context);
    }
    std::cout << std::endl;
#endif
}

void StructuredLogger::log_error(const std::string& message,
                                const std::unordered_map<std::string, std::string>& context) {
    if (!initialized_) return;

#ifdef SPDLOG_ACTIVE_LEVEL
    if (context.empty()) {
        console_logger_->error(message);
        file_logger_->error(message);
    } else {
        std::string formatted_msg = message + " " + format_json_context(context);
        console_logger_->error(formatted_msg);
        file_logger_->error(formatted_msg);
    }
#else
    std::cerr << "[ERROR] " << message;
    if (!context.empty()) {
        std::cerr << " " << format_json_context(context);
    }
    std::cerr << std::endl;
#endif
}

void StructuredLogger::log_timing(const std::string& operation,
                                 double duration_ms,
                                 const std::string& session_uuid) {
    log_info("Timing", {
        {"operation", operation},
        {"duration_ms", std::to_string(duration_ms)},
        {"session_uuid", session_uuid}
    });
}

void StructuredLogger::log_memory_usage(size_t memory_mb,
                                       const std::string& component,
                                       const std::string& session_uuid) {
    log_info("Memory usage", {
        {"component", component},
        {"memory_mb", std::to_string(memory_mb)},
        {"session_uuid", session_uuid}
    });
}

void StructuredLogger::log_system_info() {
    if (!system_info_cached_) {
        system_info_ = collect_system_info();
        system_info_cached_ = true;
    }

    log_info("System information", {
        {"cpu_model", system_info_.cpu_model},
        {"total_memory_gb", std::to_string(system_info_.total_memory_gb)},
        {"os_version", system_info_.os_version},
        {"build_config", system_info_.build_config},
        {"git_commit", system_info_.git_commit}
    });

    for (const auto& provider : system_info_.available_providers) {
        log_info("Available provider", {{"provider", provider}});
    }
}

void StructuredLogger::log_gpu_diagnostics() {
    // This would typically query GPU info via CUDA/HIP/OpenCL
    log_info("GPU diagnostics requested - implementation depends on available backends");
}

void StructuredLogger::log_model_load_event(const std::string& model_name,
                                           const std::string& model_path,
                                           bool success,
                                           const std::string& error_msg) {
    if (success) {
        log_info("Model loaded successfully", {
            {"model_name", model_name},
            {"model_path", model_path}
        });
    } else {
        log_error("Model load failed", {
            {"model_name", model_name},
            {"model_path", model_path},
            {"error", error_msg}
        });
    }
}

std::vector<SessionMetrics> StructuredLogger::get_session_history(
    const std::string& model_name,
    const std::chrono::hours& time_window) const {

    std::lock_guard<std::mutex> lock(logger_mutex_);

    auto cutoff_time = std::chrono::system_clock::now() - time_window;
    std::vector<SessionMetrics> filtered;

    for (const auto& metrics : session_history_) {
        // Filter by time window
        if (metrics.timestamp < cutoff_time) continue;

        // Filter by model name if specified
        if (!model_name.empty() && metrics.model_name != model_name) continue;

        filtered.push_back(metrics);
    }

    return filtered;
}

void StructuredLogger::set_log_level(LogLevel level) {
    current_level_ = level;

#ifdef SPDLOG_ACTIVE_LEVEL
    if (console_logger_) {
        console_logger_->set_level(static_cast<spdlog::level::level_enum>(level));
    }
    if (file_logger_) {
        file_logger_->set_level(static_cast<spdlog::level::level_enum>(level));
    }
#endif
}

void StructuredLogger::enable_console_logging(bool enable) {
#ifdef SPDLOG_ACTIVE_LEVEL
    if (console_logger_) {
        console_logger_->set_level(enable ?
            static_cast<spdlog::level::level_enum>(current_level_) :
            spdlog::level::off);
    }
#endif
}

void StructuredLogger::enable_file_logging(bool enable) {
#ifdef SPDLOG_ACTIVE_LEVEL
    if (file_logger_) {
        file_logger_->set_level(enable ?
            spdlog::level::debug :
            spdlog::level::off);
    }
#endif
}

void StructuredLogger::enable_json_logging(bool enable) {
#ifdef SPDLOG_ACTIVE_LEVEL
    if (json_logger_) {
        json_logger_->set_level(enable ?
            spdlog::level::trace :
            spdlog::level::off);
    }
#endif
}

void StructuredLogger::flush() {
#ifdef SPDLOG_ACTIVE_LEVEL
    if (console_logger_) console_logger_->flush();
    if (file_logger_) file_logger_->flush();
    if (json_logger_) json_logger_->flush();
#endif
}

void StructuredLogger::shutdown() {
    if (!initialized_) return;

    std::lock_guard<std::mutex> lock(logger_mutex_);

    log_info("StructuredLogger shutting down");

    flush();

#ifdef SPDLOG_ACTIVE_LEVEL
    spdlog::shutdown();
#endif

    initialized_ = false;
}

std::string StructuredLogger::generate_session_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);

    std::string uuid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";
    for (auto& c : uuid) {
        if (c == 'x') {
            c = "0123456789abcdef"[dis(gen)];
        } else if (c == 'y') {
            c = "89ab"[dis(gen) & 3];
        }
    }
    return uuid;
}

std::string StructuredLogger::get_timestamp_iso8601() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';

    return oss.str();
}

SystemInfo StructuredLogger::collect_system_info() {
    SystemInfo info;

#ifdef __linux__
    // Get CPU information
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                info.cpu_model = line.substr(pos + 2);
                break;
            }
        }
    }

    // Get memory information
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            std::istringstream iss(line);
            std::string token;
            size_t mem_kb = 0;
            iss >> token >> mem_kb; // Skip "MemTotal:" and get value
            info.total_memory_gb = mem_kb / (1024 * 1024);
            break;
        }
    }

    // Get OS version
    struct utsname sys_info;
    if (uname(&sys_info) == 0) {
        info.os_version = std::string(sys_info.sysname) + " " +
                         std::string(sys_info.release) + " " +
                         std::string(sys_info.machine);
    }
#endif

#ifdef _WIN32
    // Windows system info collection
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    info.cpu_model = "Windows CPU"; // Placeholder

    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem_info)) {
        info.total_memory_gb = mem_info.ullTotalPhys / (1024 * 1024 * 1024);
    }

    info.os_version = "Windows"; // Placeholder
#endif

    // Build configuration
#ifdef NDEBUG
    info.build_config = "Release";
#else
    info.build_config = "Debug";
#endif

    // Git commit (placeholder - would typically be filled by build system)
    info.git_commit = "unknown";

    // Available providers (placeholder - would be filled by actual detection)
    info.available_providers = {"CPU"};

    return info;
}

std::string StructuredLogger::format_json_context(const std::unordered_map<std::string, std::string>& context) {
    if (context.empty()) return "{}";

    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [key, value] : context) {
        if (!first) oss << ",";
        oss << "\"" << key << "\":\"" << value << "\"";
        first = false;
    }
    oss << "}";
    return oss.str();
}

} // namespace cv_stereo
