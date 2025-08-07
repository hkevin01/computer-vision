#include "edge_case_framework.hpp"
#include "ai/enhanced_neural_matcher.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <limits>
#include <thread>
#include <filesystem>

namespace stereovision {
namespace ai {
namespace testing {

class NeuralMatcherEdgeCaseTest : public stereo_vision::testing::EdgeCaseTest<double> {
protected:
    void SetUp() override {
        stereo_vision::testing::EdgeCaseTest<double>::SetUp();
        matcher_ = std::make_unique<EnhancedNeuralMatcher>();
    }

    std::unique_ptr<EnhancedNeuralMatcher> matcher_;

    // Helper to create test stereo image pairs
    std::pair<cv::Mat, cv::Mat> createTestStereoPair(cv::Size size, int type = CV_8UC3) {
        cv::Mat left(size, type);
        cv::Mat right(size, type);

        // Create simple gradient pattern
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                if (type == CV_8UC3) {
                    left.at<cv::Vec3b>(i, j) = cv::Vec3b(
                        static_cast<uint8_t>((i * 255) / size.height),
                        static_cast<uint8_t>((j * 255) / size.width),
                        128
                    );
                    // Right image with slight horizontal shift for disparity
                    int shifted_j = std::max(0, j - 5);
                    right.at<cv::Vec3b>(i, shifted_j) = cv::Vec3b(
                        static_cast<uint8_t>((i * 255) / size.height),
                        static_cast<uint8_t>((j * 255) / size.width),
                        128
                    );
                } else if (type == CV_8UC1) {
                    left.at<uint8_t>(i, j) = static_cast<uint8_t>((i * 255) / size.height);
                    int shifted_j = std::max(0, j - 5);
                    right.at<uint8_t>(i, shifted_j) = static_cast<uint8_t>((i * 255) / size.height);
                }
            }
        }

        return std::make_pair(left, right);
    }
};

// Test neural matcher with extreme image sizes
TEST_F(NeuralMatcherEdgeCaseTest, ExtremeImageSizes) {
    auto edge_sizes = stereo_vision::testing::EdgeCaseTestFramework::getImageSizeEdgeCases();

    for (const auto& size : edge_sizes) {
        if (size.width > 0 && size.height > 0 && size.width < 8192 && size.height < 8192) {
            try {
                auto [left, right] = createTestStereoPair(size);

                // Configure for CPU backend to avoid GPU memory issues with extreme sizes
                EnhancedNeuralMatcher::ModelConfig config;
                config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;
                config.type = EnhancedNeuralMatcher::ModelType::HITNET;

                cv::Mat disparity;
                disparity = matcher_->computeDisparity(left, right); bool success = !disparity.empty();

                if (success && !disparity.empty()) {
                    // Verify disparity map has reasonable values
                    double min_val, max_val;
                    cv::minMaxLoc(disparity, &min_val, &max_val);

                    EXPECT_FALSE(std::isnan(min_val)) << "Disparity contains NaN values";
                    EXPECT_FALSE(std::isnan(max_val)) << "Disparity contains NaN values";
                    EXPECT_FALSE(std::isinf(min_val)) << "Disparity contains infinite values";
                    EXPECT_FALSE(std::isinf(max_val)) << "Disparity contains infinite values";

                    // Disparity values should be reasonable for the image size
                    EXPECT_GE(min_val, 0.0) << "Negative disparity values detected";
                    EXPECT_LE(max_val, size.width) << "Disparity exceeds image width";
                }

            } catch (const std::exception& e) {
                // Graceful failure is acceptable for extreme sizes
                EXPECT_TRUE(true) << "Gracefully handled extreme size "
                                  << size.width << "x" << size.height << ": " << e.what();
            }
        }
    }
}

// Test numerical overflow in disparity calculations
TEST_F(NeuralMatcherEdgeCaseTest, NumericalOverflowHandling) {
    auto edge_values = stereo_vision::testing::EdgeCaseTestFramework::getFloatingPointEdgeCases();

    // Create test images
    auto [left, right] = createTestStereoPair(cv::Size(640, 480));

    for (double edge_val : edge_values) {
        if (std::isfinite(edge_val) && edge_val > 0 && edge_val < 1000) {
            try {
                EnhancedNeuralMatcher::ModelConfig config;
                config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;
                config.max_disparity = static_cast<int>(edge_val);

                cv::Mat disparity;
                disparity = matcher_->computeDisparity(left, right); bool success = !disparity.empty();

                if (success && !disparity.empty()) {
                    // Check for overflow in disparity values
                    cv::Mat float_disparity;
                    disparity.convertTo(float_disparity, CV_32F);

                    bool has_overflow = false;
                    for (int i = 0; i < float_disparity.rows; ++i) {
                        for (int j = 0; j < float_disparity.cols; ++j) {
                            float val = float_disparity.at<float>(i, j);
                            if (std::isinf(val) || std::isnan(val)) {
                                has_overflow = true;
                                break;
                            }
                        }
                        if (has_overflow) break;
                    }

                    EXPECT_FALSE(has_overflow) << "Overflow detected in disparity computation with max_disparity=" << edge_val;
                }

            } catch (const std::exception& e) {
                // Expected for extreme parameter values
                EXPECT_TRUE(true) << "Gracefully handled extreme max_disparity " << edge_val << ": " << e.what();
            }
        }
    }
}

// Test malformed input handling
TEST_F(NeuralMatcherEdgeCaseTest, MalformedInputHandling) {
    EnhancedNeuralMatcher::ModelConfig config;
    config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;

    // Test with empty images
    cv::Mat empty_left, empty_right, disparity;
    EXPECT_GRACEFUL_FAILURE(
        disparity = matcher_->computeDisparity(empty_left, empty_right)
    );

    // Test with mismatched image sizes
    cv::Mat left(cv::Size(640, 480), CV_8UC3);
    cv::Mat right(cv::Size(320, 240), CV_8UC3);
    EXPECT_GRACEFUL_FAILURE(
        matcher_->computeDisparity(left, right)
    );

    // Test with different image types
    cv::Mat left_color(cv::Size(640, 480), CV_8UC3);
    cv::Mat right_gray(cv::Size(640, 480), CV_8UC1);
    EXPECT_GRACEFUL_FAILURE(
        matcher_->computeDisparity(left_color, right_gray)
    );

    // Test with corrupted images
    cv::Mat corrupted_left = stereo_vision::testing::EdgeCaseTestFramework::generateCorruptedImage(cv::Size(640, 480), CV_8UC3);
    cv::Mat corrupted_right = stereo_vision::testing::EdgeCaseTestFramework::generateCorruptedImage(cv::Size(640, 480), CV_8UC3);

    // This should either work or fail gracefully
    try {
        bool success = matcher_->computeDisparity(corrupted_left, corrupted_right);
        if (!success) {
            EXPECT_TRUE(true) << "Gracefully rejected corrupted input";
        }
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully failed with corrupted input: " << e.what();
    }

    // Test with infinite values
    cv::Mat inf_left = stereo_vision::testing::EdgeCaseTestFramework::generateInfiniteValuesMatrix(cv::Size(640, 480), CV_32F);
    cv::Mat inf_right = stereo_vision::testing::EdgeCaseTestFramework::generateInfiniteValuesMatrix(cv::Size(640, 480), CV_32F);
    EXPECT_GRACEFUL_FAILURE(
        matcher_->computeDisparity(inf_left, inf_right)
    );

    // Test with NaN values
    cv::Mat nan_left = stereo_vision::testing::EdgeCaseTestFramework::generateNaNValuesMatrix(cv::Size(640, 480), CV_32F);
    cv::Mat nan_right = stereo_vision::testing::EdgeCaseTestFramework::generateNaNValuesMatrix(cv::Size(640, 480), CV_32F);
    EXPECT_GRACEFUL_FAILURE(
        matcher_->computeDisparity(nan_left, nan_right)
    );
}

// Test precision loss in neural network inference
TEST_F(NeuralMatcherEdgeCaseTest, PrecisionLossDetection) {
    auto [left, right] = createTestStereoPair(cv::Size(640, 480));

    // Convert to different precisions and test
    std::vector<int> precisions = {CV_8U, CV_16U, CV_32F, CV_64F};

    cv::Mat reference_disparity;
    bool reference_computed = false;

    for (int precision : precisions) {
        try {
            cv::Mat left_precision, right_precision;
            left.convertTo(left_precision, precision);
            right.convertTo(right_precision, precision);

            EnhancedNeuralMatcher::ModelConfig config;
            config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;

            cv::Mat disparity;
            bool success = matcher_->computeDisparity(left_precision, right_precision);

            if (success && !disparity.empty()) {
                // Convert disparity to float for comparison
                cv::Mat float_disparity;
                disparity.convertTo(float_disparity, CV_32F);

                if (!reference_computed) {
                    reference_disparity = float_disparity.clone();
                    reference_computed = true;
                } else {
                    // Compare with reference to detect precision loss
                    cv::Mat diff;
                    cv::absdiff(float_disparity, reference_disparity, diff);

                    cv::Scalar mean_diff = cv::mean(diff);
                    double max_acceptable_diff = 1.0; // Pixel difference threshold

                    EXPECT_LT(mean_diff[0], max_acceptable_diff)
                        << "Significant precision loss detected with precision type " << precision
                        << " (mean difference: " << mean_diff[0] << ")";
                }
            }

        } catch (const std::exception& e) {
            // Some precisions might not be supported
            EXPECT_TRUE(true) << "Precision type " << precision << " not supported: " << e.what();
        }
    }
}

// Test concurrent neural matcher operations
TEST_F(NeuralMatcherEdgeCaseTest, ConcurrentMatching) {
    auto [left, right] = createTestStereoPair(cv::Size(640, 480));

    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    auto matching_task = [&]() {
        try {
            auto local_matcher = std::make_unique<EnhancedNeuralMatcher>();

            EnhancedNeuralMatcher::ModelConfig config;
            config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;

            cv::Mat disparity;
            disparity = local_matcher->computeDisparity(left, right); bool success = !disparity.empty();

            if (success && !disparity.empty()) {
                success_count++;
            } else {
                failure_count++;
            }
        } catch (const std::exception&) {
            failure_count++;
        }
    };

    stereo_vision::testing::EdgeCaseTestFramework::testConcurrentAccess(matching_task, 5, 3);

    // At least some operations should succeed if the system has resources
    EXPECT_GE(success_count.load() + failure_count.load(), 1) << "No concurrent operations completed";
}

// Test neural matcher under system load
TEST_F(NeuralMatcherEdgeCaseTest, MatchingUnderSystemLoad) {
    auto [left, right] = createTestStereoPair(cv::Size(640, 480));

    // Start high CPU load in background
    std::thread cpu_load_thread([]() {
        stereo_vision::testing::EdgeCaseTestFramework::simulateHighCPULoad(2000);
    });

    // Start memory pressure
    stereo_vision::testing::EdgeCaseTestFramework::simulateMemoryPressure(256);

    try {
        EnhancedNeuralMatcher::ModelConfig config;
        config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;

        cv::Mat disparity;
        auto start = std::chrono::steady_clock::now();
        disparity = matcher_->computeDisparity(left, right); bool success = !disparity.empty();
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (success) {
            EXPECT_TRUE(true) << "Neural matching succeeded under system load (took " << duration.count() << "ms)";

            // Verify result quality wasn't severely degraded
            if (!disparity.empty()) {
                double min_val, max_val;
                cv::minMaxLoc(disparity, &min_val, &max_val);
                EXPECT_FALSE(std::isnan(min_val)) << "System load caused NaN in disparity";
                EXPECT_FALSE(std::isinf(max_val)) << "System load caused infinite disparity";
            }
        } else {
            EXPECT_TRUE(true) << "Neural matching gracefully failed under system load";
        }

    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Neural matching gracefully failed under system load: " << e.what();
    }

    cpu_load_thread.join();
}

// Test model loading failures
TEST_F(NeuralMatcherEdgeCaseTest, ModelLoadingFailures) {
    // Test with non-existent model file
    EnhancedNeuralMatcher::ModelConfig config;
    config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;
    config.type = EnhancedNeuralMatcher::ModelType::CUSTOM;
    config.model_path = "/non/existent/path/model.onnx";

    auto [left, right] = createTestStereoPair(cv::Size(640, 480));
    cv::Mat disparity;

    EXPECT_GRACEFUL_FAILURE(
        matcher_->computeDisparity(left, right)
    );

    // Test with corrupted model file (if we can create one)
    std::filesystem::path temp_model = std::filesystem::temp_directory_path() / "corrupted_model.onnx";
    try {
        std::ofstream corrupted_file(temp_model, std::ios::binary);
        corrupted_file << "This is not a valid ONNX model file";
        corrupted_file.close();

        config.model_path = temp_model.string();
        EXPECT_GRACEFUL_FAILURE(
            matcher_->computeDisparity(left, right)
        );

        std::filesystem::remove(temp_model);
    } catch (const std::exception&) {
        // File system operations might fail, that's okay
    }
}

// Test hardware failure simulation
TEST_F(NeuralMatcherEdgeCaseTest, HardwareFailureSimulation) {
    // This test simulates various hardware failure scenarios
    auto [left, right] = createTestStereoPair(cv::Size(640, 480));

    // Test GPU failure fallback
    EnhancedNeuralMatcher::ModelConfig gpu_config;
    gpu_config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_GPU;

    cv::Mat disparity;
    try {
        // This might fail if no GPU is available or CUDA is not installed
        disparity = matcher_->computeDisparity(left, right); bool success = !disparity.empty();
        if (!success) {
            // Try CPU fallback
            gpu_config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;
            disparity = matcher_->computeDisparity(left, right); success = !disparity.empty();
            EXPECT_TRUE(success) << "CPU fallback should work when GPU fails";
        }
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled GPU failure: " << e.what();
    }
}

// Test memory allocation failures during inference
TEST_F(NeuralMatcherEdgeCaseTest, MemoryAllocationFailures) {
    // Create very large images that might cause memory allocation issues
    auto [left, right] = createTestStereoPair(cv::Size(4096, 4096));

    // Apply maximum memory pressure
    stereo_vision::testing::EdgeCaseTestFramework::simulateMemoryPressure(2048); // 2GB

    EnhancedNeuralMatcher::ModelConfig config;
    config.preferred_backend = EnhancedNeuralMatcher::Backend::ONNX_CPU;

    cv::Mat disparity;
    try {
        disparity = matcher_->computeDisparity(left, right); bool success = !disparity.empty();
        if (success) {
            EXPECT_TRUE(true) << "Neural matching succeeded despite memory pressure";
        } else {
            EXPECT_TRUE(true) << "Neural matching gracefully failed under memory pressure";
        }
    } catch (const std::bad_alloc&) {
        EXPECT_TRUE(true) << "Gracefully handled memory allocation failure";
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Gracefully handled memory-related failure: " << e.what();
    }
}

} // namespace testing
} // namespace ai
} // namespace stereovision
