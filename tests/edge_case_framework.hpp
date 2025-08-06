#pragma once

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <limits>
#include <random>
#include <thread>
#include <chrono>

namespace stereo_vision {
namespace testing {

/**
 * @brief Comprehensive edge case testing framework
 *
 * This framework provides utilities for testing:
 * - Numerical edge cases (overflow, underflow, precision loss)
 * - Memory constraints and allocation failures
 * - Hardware simulation failures
 * - Malformed input handling
 * - System load conditions
 * - Concurrent access scenarios
 */
class EdgeCaseTestFramework {
public:
    // Numerical edge case generators
    static std::vector<double> getFloatingPointEdgeCases();
    static std::vector<int> getIntegerEdgeCases();
    static std::vector<cv::Size> getImageSizeEdgeCases();

    // Memory stress testing
    static void simulateMemoryPressure(size_t mb_to_allocate = 1024);
    static void clearMemoryPressure();

    // Hardware failure simulation
    static void simulateGPUFailure();
    static void simulateFileSystemFailure();
    static void simulateNetworkFailure();

    // Malformed input generators
    static cv::Mat generateCorruptedImage(cv::Size size, int type);
    static cv::Mat generateInfiniteValuesMatrix(cv::Size size, int type);
    static cv::Mat generateNaNValuesMatrix(cv::Size size, int type);

    // System load simulation
    static void simulateHighCPULoad(int duration_ms = 1000);
    static void simulateHighMemoryLoad(int duration_ms = 1000);

    // Concurrent access testing
    template<typename Func>
    static void testConcurrentAccess(Func func, int thread_count = 10, int iterations = 100);

    // Precision testing utilities
    static bool isWithinTolerance(double actual, double expected, double tolerance = 1e-9);
    static bool hasSignificantPrecisionLoss(double original, double computed);

private:
    static std::vector<void*> allocated_memory_;
    static std::mt19937 rng_;
};

/**
 * @brief Parameterized test fixture for edge cases
 */
template<typename T>
class EdgeCaseTest : public ::testing::TestWithParam<T> {
protected:
    void SetUp() override {
        // Initialize edge case testing environment
        EdgeCaseTestFramework::clearMemoryPressure();
    }

    void TearDown() override {
        // Clean up after edge case testing
        EdgeCaseTestFramework::clearMemoryPressure();
    }
};

/**
 * @brief Macros for edge case testing
 */
#define EXPECT_NO_OVERFLOW(expr) \
    do { \
        auto start_val = expr; \
        EXPECT_FALSE(std::isinf(start_val)) << "Expression resulted in overflow"; \
        EXPECT_FALSE(std::isnan(start_val)) << "Expression resulted in NaN"; \
    } while(0)

#define EXPECT_GRACEFUL_FAILURE(expr) \
    do { \
        bool exception_caught = false; \
        try { \
            expr; \
        } catch (const std::exception& e) { \
            exception_caught = true; \
            EXPECT_TRUE(true) << "Gracefully handled exception: " << e.what(); \
        } \
        if (!exception_caught) { \
            EXPECT_TRUE(false) << "Expected graceful failure but none occurred"; \
        } \
    } while(0)

#define EXPECT_ROBUST_OPERATION(expr, expected_result) \
    do { \
        auto result = expr; \
        EXPECT_EQ(result, expected_result) << "Operation failed under stress conditions"; \
    } while(0)

} // namespace testing
} // namespace stereo_vision
