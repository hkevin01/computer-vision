#include "edge_case_framework.hpp"
#include <algorithm>
#include <thread>
#include <vector>
#include <random>

namespace stereo_vision {
namespace testing {

std::vector<void*> EdgeCaseTestFramework::allocated_memory_;
std::mt19937 EdgeCaseTestFramework::rng_(std::chrono::steady_clock::now().time_since_epoch().count());

std::vector<double> EdgeCaseTestFramework::getFloatingPointEdgeCases() {
    return {
        // Basic edge cases
        0.0, -0.0,
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::epsilon(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::signaling_NaN(),

        // Precision boundary cases
        1.0 + std::numeric_limits<double>::epsilon(),
        1.0 - std::numeric_limits<double>::epsilon(),

        // Common problematic values
        1e-308, 1e308, -1e308, 1e-16, 1e16,

        // Values that can cause overflow in common operations
        std::sqrt(std::numeric_limits<double>::max()),
        std::numeric_limits<double>::max() / 2.0,

        // Denormalized numbers
        std::numeric_limits<double>::denorm_min(),

        // Values near zero that might cause division issues
        1e-100, -1e-100, 1e-200, -1e-200
    };
}

std::vector<int> EdgeCaseTestFramework::getIntegerEdgeCases() {
    return {
        0, 1, -1,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::min() + 1,
        std::numeric_limits<int>::max() - 1,

        // Powers of 2 (common in computer graphics)
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,

        // Negative powers of 2
        -1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024,

        // Common image/video dimensions edge cases
        1920, 1080, 4096, 8192, 16384
    };
}

std::vector<cv::Size> EdgeCaseTestFramework::getImageSizeEdgeCases() {
    return {
        // Minimal sizes
        cv::Size(1, 1), cv::Size(0, 0), cv::Size(1, 0), cv::Size(0, 1),

        // Very small sizes
        cv::Size(2, 2), cv::Size(3, 3), cv::Size(4, 4),

        // Extreme aspect ratios
        cv::Size(1, 10000), cv::Size(10000, 1),
        cv::Size(1, 8192), cv::Size(8192, 1),

        // Large but reasonable sizes
        cv::Size(8192, 8192), cv::Size(4096, 4096),
        cv::Size(16384, 16384),

        // Common problematic sizes
        cv::Size(1920, 1080), cv::Size(3840, 2160), // 4K
        cv::Size(7680, 4320), // 8K

        // Odd sizes that might cause alignment issues
        cv::Size(1919, 1079), cv::Size(1921, 1081),
        cv::Size(1023, 767), cv::Size(2047, 1535)
    };
}

void EdgeCaseTestFramework::simulateMemoryPressure(size_t mb_to_allocate) {
    try {
        for (size_t i = 0; i < mb_to_allocate; ++i) {
            void* ptr = std::malloc(1024 * 1024); // 1 MB
            if (ptr) {
                allocated_memory_.push_back(ptr);
                // Touch the memory to ensure it's actually allocated
                std::memset(ptr, 0x42, 1024 * 1024);
            }
        }
    } catch (const std::bad_alloc&) {
        // Expected behavior under memory pressure
    }
}

void EdgeCaseTestFramework::clearMemoryPressure() {
    for (void* ptr : allocated_memory_) {
        std::free(ptr);
    }
    allocated_memory_.clear();
}

cv::Mat EdgeCaseTestFramework::generateCorruptedImage(cv::Size size, int type) {
    cv::Mat corrupted(size, type);

    // Fill with random garbage data
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (int i = 0; i < corrupted.rows; ++i) {
        for (int j = 0; j < corrupted.cols; ++j) {
            for (int c = 0; c < corrupted.channels(); ++c) {
                corrupted.ptr<uint8_t>(i)[j * corrupted.channels() + c] = dist(rng_);
            }
        }
    }

    // Introduce some systematic corruption
    if (size.height > 10 && size.width > 10) {
        // Create "dead pixels" patterns
        for (int i = 0; i < size.height; i += 7) {
            for (int j = 0; j < size.width; j += 11) {
                if (type == CV_8UC1) {
                    corrupted.at<uint8_t>(i, j) = 0;
                } else if (type == CV_8UC3) {
                    corrupted.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 255); // Magenta
                }
            }
        }
    }

    return corrupted;
}

cv::Mat EdgeCaseTestFramework::generateInfiniteValuesMatrix(cv::Size size, int type) {
    cv::Mat inf_mat(size, type);

    if (type == CV_64F) {
        inf_mat.setTo(std::numeric_limits<double>::infinity());
        // Mix some negative infinities
        for (int i = 0; i < size.height; i += 2) {
            for (int j = 0; j < size.width; j += 2) {
                inf_mat.at<double>(i, j) = -std::numeric_limits<double>::infinity();
            }
        }
    } else if (type == CV_32F) {
        inf_mat.setTo(std::numeric_limits<float>::infinity());
        for (int i = 0; i < size.height; i += 2) {
            for (int j = 0; j < size.width; j += 2) {
                inf_mat.at<float>(i, j) = -std::numeric_limits<float>::infinity();
            }
        }
    }

    return inf_mat;
}

cv::Mat EdgeCaseTestFramework::generateNaNValuesMatrix(cv::Size size, int type) {
    cv::Mat nan_mat(size, type);

    if (type == CV_64F) {
        nan_mat.setTo(std::numeric_limits<double>::quiet_NaN());
        // Mix some signaling NaNs if supported
        for (int i = 1; i < size.height; i += 3) {
            for (int j = 1; j < size.width; j += 3) {
                nan_mat.at<double>(i, j) = std::numeric_limits<double>::signaling_NaN();
            }
        }
    } else if (type == CV_32F) {
        nan_mat.setTo(std::numeric_limits<float>::quiet_NaN());
        for (int i = 1; i < size.height; i += 3) {
            for (int j = 1; j < size.width; j += 3) {
                nan_mat.at<float>(i, j) = std::numeric_limits<float>::signaling_NaN();
            }
        }
    }

    return nan_mat;
}

void EdgeCaseTestFramework::simulateHighCPULoad(int duration_ms) {
    auto start = std::chrono::steady_clock::now();
    auto end = start + std::chrono::milliseconds(duration_ms);

    // Spawn threads to consume CPU
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;

    std::atomic<bool> stop_flag{false};

    for (unsigned int i = 0; i < num_threads; ++i) {
        workers.emplace_back([&stop_flag]() {
            volatile double dummy = 0.0;
            while (!stop_flag) {
                // Meaningless computation to consume CPU
                dummy += std::sin(dummy) * std::cos(dummy);
            }
        });
    }

    // Wait for the specified duration
    std::this_thread::sleep_until(end);
    stop_flag = true;

    // Clean up threads
    for (auto& worker : workers) {
        worker.join();
    }
}

void EdgeCaseTestFramework::simulateHighMemoryLoad(int duration_ms) {
    auto start = std::chrono::steady_clock::now();
    auto end = start + std::chrono::milliseconds(duration_ms);

    std::vector<std::vector<uint8_t>> memory_hogs;

    try {
        while (std::chrono::steady_clock::now() < end) {
            // Allocate 10MB chunks
            memory_hogs.emplace_back(10 * 1024 * 1024, 0x42);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (const std::bad_alloc&) {
        // Expected under memory pressure
    }

    // Memory will be automatically freed when memory_hogs goes out of scope
}

bool EdgeCaseTestFramework::isWithinTolerance(double actual, double expected, double tolerance) {
    if (std::isnan(actual) && std::isnan(expected)) {
        return true; // Both NaN
    }
    if (std::isinf(actual) && std::isinf(expected)) {
        return (actual > 0) == (expected > 0); // Same sign infinity
    }
    if (std::isnan(actual) || std::isnan(expected) || std::isinf(actual) || std::isinf(expected)) {
        return false;
    }

    return std::abs(actual - expected) <= tolerance;
}

bool EdgeCaseTestFramework::hasSignificantPrecisionLoss(double original, double computed) {
    if (original == 0.0) {
        return computed != 0.0;
    }

    double relative_error = std::abs((computed - original) / original);
    return relative_error > 1e-12; // Threshold for significant precision loss
}

} // namespace testing
} // namespace stereo_vision
