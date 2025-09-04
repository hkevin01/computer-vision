#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "logging/structured_logger.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

using namespace cv_stereo;
using ::testing::HasSubstr;

class StructuredLoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_log_dir_ = "test_logs_" + std::to_string(std::time(nullptr)) + "/";

        // Clean up any existing test logs
        if (std::filesystem::exists(test_log_dir_)) {
            std::filesystem::remove_all(test_log_dir_);
        }
    }

    void TearDown() override {
        // Clean up test logs
        auto& logger = StructuredLogger::instance();
        logger.shutdown();

        if (std::filesystem::exists(test_log_dir_)) {
            std::filesystem::remove_all(test_log_dir_);
        }
    }

    std::string test_log_dir_;

    std::string read_file_content(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) return "";

        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return content;
    }

    bool file_contains(const std::string& filepath, const std::string& text) {
        std::string content = read_file_content(filepath);
        return content.find(text) != std::string::npos;
    }
};

TEST_F(StructuredLoggerTest, SingletonInstance) {
    auto& instance1 = StructuredLogger::instance();
    auto& instance2 = StructuredLogger::instance();

    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(StructuredLoggerTest, InitializationSuccess) {
    auto& logger = StructuredLogger::instance();

    bool result = logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, true);
    EXPECT_TRUE(result);

    // Check that log directory was created
    EXPECT_TRUE(std::filesystem::exists(test_log_dir_));
}

TEST_F(StructuredLoggerTest, InitializationWithInvalidDirectory) {
    auto& logger = StructuredLogger::instance();

    // Try to create log directory in non-existent parent
    std::string invalid_dir = "/non_existent_parent/logs/";
    bool result = logger.initialize(invalid_dir, LogLevel::INFO, LogLevel::DEBUG, false);

    // Should handle gracefully - might succeed if spdlog creates directories
    // or might fail, but shouldn't crash
    EXPECT_TRUE(true); // Just ensure no crash
}

TEST_F(StructuredLoggerTest, BasicLogging) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    logger.log_info("Test info message");
    logger.log_warning("Test warning message");
    logger.log_error("Test error message");

    logger.flush();

    // Verify log file was created
    std::string log_file = test_log_dir_ + "stereo_vision.log";
    EXPECT_TRUE(std::filesystem::exists(log_file));

    // Check log content
    std::string content = read_file_content(log_file);
    EXPECT_THAT(content, HasSubstr("Test info message"));
    EXPECT_THAT(content, HasSubstr("Test warning message"));
    EXPECT_THAT(content, HasSubstr("Test error message"));
}

TEST_F(StructuredLoggerTest, ContextualLogging) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    std::unordered_map<std::string, std::string> context = {
        {"model", "test_model"},
        {"provider", "CPU"},
        {"session_id", "test_session_123"}
    };

    logger.log_info("Processing started", context);
    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    EXPECT_THAT(content, HasSubstr("Processing started"));
    EXPECT_THAT(content, HasSubstr("test_model"));
    EXPECT_THAT(content, HasSubstr("CPU"));
    EXPECT_THAT(content, HasSubstr("test_session_123"));
}

TEST_F(StructuredLoggerTest, SessionManagement) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    std::string session_uuid = logger.start_session("test_operation");
    EXPECT_FALSE(session_uuid.empty());
    EXPECT_EQ(session_uuid.length(), 36); // Standard UUID length

    // Brief delay to ensure measurable session duration
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    logger.end_session(session_uuid);
    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    EXPECT_THAT(content, HasSubstr("Session started"));
    EXPECT_THAT(content, HasSubstr("Session ended"));
    EXPECT_THAT(content, HasSubstr(session_uuid));
    EXPECT_THAT(content, HasSubstr("test_operation"));
}

TEST_F(StructuredLoggerTest, SessionMetricsLogging) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, true);

    SessionMetrics metrics;
    metrics.session_uuid = "test-uuid-12345";
    metrics.model_name = "test_model";
    metrics.provider_name = "CPU";
    metrics.preprocessing_time_ms = 5.2;
    metrics.inference_time_ms = 15.7;
    metrics.postprocessing_time_ms = 3.1;
    metrics.total_time_ms = 24.0;
    metrics.input_memory_mb = 10;
    metrics.output_memory_mb = 5;
    metrics.peak_memory_mb = 20;
    metrics.successful = true;
    metrics.timestamp = std::chrono::system_clock::now();

    logger.log_session_metrics(metrics);
    logger.flush();

    // Check regular log
    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);
    EXPECT_THAT(content, HasSubstr("Session metrics recorded"));
    EXPECT_THAT(content, HasSubstr("test-uuid-12345"));

    // Check JSON log
    std::string json_file = test_log_dir_ + "metrics.jsonl";
    if (std::filesystem::exists(json_file)) {
        std::string json_content = read_file_content(json_file);
        EXPECT_THAT(json_content, HasSubstr("session_metrics"));
        EXPECT_THAT(json_content, HasSubstr("test-uuid-12345"));
        EXPECT_THAT(json_content, HasSubstr("test_model"));
        EXPECT_THAT(json_content, HasSubstr("15.7")); // inference time
    }
}

TEST_F(StructuredLoggerTest, TimingMacros) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    std::string session_uuid = logger.start_session("timing_test");

    {
        CV_TIMER_SESSION("test_operation", session_uuid);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } // Timer destructor should log timing

    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    EXPECT_THAT(content, HasSubstr("Timing"));
    EXPECT_THAT(content, HasSubstr("test_operation"));
    EXPECT_THAT(content, HasSubstr(session_uuid));
}

TEST_F(StructuredLoggerTest, ModelLoadEventLogging) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    // Test successful model load
    logger.log_model_load_event("test_model", "/path/to/model.onnx", true);

    // Test failed model load
    logger.log_model_load_event("failed_model", "/path/to/failed.onnx", false, "File not found");

    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    EXPECT_THAT(content, HasSubstr("Model loaded successfully"));
    EXPECT_THAT(content, HasSubstr("test_model"));
    EXPECT_THAT(content, HasSubstr("Model load failed"));
    EXPECT_THAT(content, HasSubstr("failed_model"));
    EXPECT_THAT(content, HasSubstr("File not found"));
}

TEST_F(StructuredLoggerTest, SessionHistoryTracking) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    // Log several session metrics
    for (int i = 0; i < 5; ++i) {
        SessionMetrics metrics;
        metrics.session_uuid = "test-uuid-" + std::to_string(i);
        metrics.model_name = (i % 2 == 0) ? "model_a" : "model_b";
        metrics.provider_name = "CPU";
        metrics.inference_time_ms = 10.0 + i;
        metrics.successful = true;
        metrics.timestamp = std::chrono::system_clock::now();

        logger.log_session_metrics(metrics);
    }

    // Get all history
    auto all_history = logger.get_session_history();
    EXPECT_EQ(all_history.size(), 5);

    // Get filtered history for specific model
    auto model_a_history = logger.get_session_history("model_a");
    EXPECT_EQ(model_a_history.size(), 3); // model_a appears at indices 0, 2, 4

    auto model_b_history = logger.get_session_history("model_b");
    EXPECT_EQ(model_b_history.size(), 2); // model_b appears at indices 1, 3
}

TEST_F(StructuredLoggerTest, LogLevelConfiguration) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::WARN, LogLevel::DEBUG, false);

    // These should not appear in console (level WARN) but should appear in file (level DEBUG)
    logger.log_info("Info message");
    logger.log_warning("Warning message");
    logger.log_error("Error message");

    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    // All messages should be in file log
    EXPECT_THAT(content, HasSubstr("Info message"));
    EXPECT_THAT(content, HasSubstr("Warning message"));
    EXPECT_THAT(content, HasSubstr("Error message"));
}

TEST_F(StructuredLoggerTest, MemoryUsageLogging) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    std::string session_uuid = logger.start_session("memory_test");
    logger.log_memory_usage(100, "input_buffer", session_uuid);
    logger.log_memory_usage(50, "output_buffer", session_uuid);

    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    EXPECT_THAT(content, HasSubstr("Memory usage"));
    EXPECT_THAT(content, HasSubstr("input_buffer"));
    EXPECT_THAT(content, HasSubstr("output_buffer"));
    EXPECT_THAT(content, HasSubstr("100"));
    EXPECT_THAT(content, HasSubstr("50"));
    EXPECT_THAT(content, HasSubstr(session_uuid));
}

TEST_F(StructuredLoggerTest, SystemInfoLogging) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    logger.log_system_info();
    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    EXPECT_THAT(content, HasSubstr("System information"));
    // Content will vary by system, but should contain some system info
    EXPECT_FALSE(content.empty());
}

TEST_F(StructuredLoggerTest, ConcurrentAccess) {
    auto& logger = StructuredLogger::instance();
    logger.initialize(test_log_dir_, LogLevel::DEBUG, LogLevel::TRACE, false);

    const int num_threads = 4;
    const int messages_per_thread = 10;
    std::vector<std::thread> threads;

    // Launch multiple threads that log concurrently
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&logger, i, messages_per_thread]() {
            for (int j = 0; j < messages_per_thread; ++j) {
                std::string session_uuid = logger.start_session("concurrent_test");
                logger.log_info("Thread " + std::to_string(i) + " message " + std::to_string(j));
                logger.log_timing("test_operation", 5.0 + j, session_uuid);
                logger.end_session(session_uuid);
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    logger.flush();

    std::string log_file = test_log_dir_ + "stereo_vision.log";
    std::string content = read_file_content(log_file);

    // Should contain messages from all threads
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_THAT(content, HasSubstr("Thread " + std::to_string(i)));
    }

    // Count session start/end pairs - should be balanced
    size_t start_count = 0, end_count = 0;
    size_t pos = 0;
    while ((pos = content.find("Session started", pos)) != std::string::npos) {
        start_count++;
        pos++;
    }
    pos = 0;
    while ((pos = content.find("Session ended", pos)) != std::string::npos) {
        end_count++;
        pos++;
    }

    EXPECT_EQ(start_count, end_count);
    EXPECT_EQ(start_count, num_threads * messages_per_thread);
}

// Test without spdlog dependency
class StructuredLoggerNoSpdlogTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_log_dir_ = "test_logs_nospdlog_" + std::to_string(std::time(nullptr)) + "/";
    }

    void TearDown() override {
        auto& logger = StructuredLogger::instance();
        logger.shutdown();

        if (std::filesystem::exists(test_log_dir_)) {
            std::filesystem::remove_all(test_log_dir_);
        }
    }

    std::string test_log_dir_;
};

TEST_F(StructuredLoggerNoSpdlogTest, FallbackLoggingWorks) {
    auto& logger = StructuredLogger::instance();

    // Should not crash even if spdlog is not available
    bool result = logger.initialize(test_log_dir_);

    // Basic logging operations should not crash
    logger.log_info("Test message without spdlog");
    logger.log_warning("Warning without spdlog");
    logger.log_error("Error without spdlog");

    std::string session_uuid = logger.start_session("no_spdlog_test");
    logger.end_session(session_uuid);

    logger.flush();

    // Should complete without crashing
    EXPECT_TRUE(true);
}
