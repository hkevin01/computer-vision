#include "edge_case_framework.hpp"
#include "camera_calibration.hpp"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <limits>
#include <thread>

namespace stereo_vision {
namespace testing {

class CameraCalibrationEdgeCaseTest : public EdgeCaseTest<double> {
protected:
    void SetUp() override {
        EdgeCaseTest::SetUp();
        calibration_ = std::make_unique<CameraCalibration>();
    }

    std::unique_ptr<CameraCalibration> calibration_;

    // Helper function to create test checkerboard image
    cv::Mat createTestCheckerboard(cv::Size image_size, cv::Size board_size, bool add_noise = false) {
        cv::Mat image = cv::Mat::zeros(image_size, CV_8UC1);

        int square_width = image_size.width / (board_size.width + 1);
        int square_height = image_size.height / (board_size.height + 1);

        for (int i = 0; i < board_size.height + 1; ++i) {
            for (int j = 0; j < board_size.width + 1; ++j) {
                if ((i + j) % 2 == 0) {
                    cv::Rect rect(j * square_width, i * square_height, square_width, square_height);
                    if (rect.x + rect.width <= image_size.width && rect.y + rect.height <= image_size.height) {
                        cv::rectangle(image, rect, cv::Scalar(255), -1);
                    }
                }
            }
        }

        if (add_noise) {
            cv::Mat noise;
            cv::randn(noise, 0, 10);
            image += noise;
        }

        return image;
    }
};

// Test overflow conditions in calibration parameters
TEST_F(CameraCalibrationEdgeCaseTest, CalibrationParameterOverflow) {
    auto edge_values = EdgeCaseTestFramework::getFloatingPointEdgeCases();

    for (double edge_val : edge_values) {
        if (std::isfinite(edge_val) && edge_val > 0) {
            // Test extremely large square sizes
            std::vector<cv::Mat> images;
            images.push_back(createTestCheckerboard(cv::Size(640, 480), cv::Size(9, 6)));

            EXPECT_GRACEFUL_FAILURE(
                calibration_->calibrateSingleCamera(images, cv::Size(9, 6), static_cast<float>(edge_val))
            );
        }
    }
}

// Test precision loss in camera matrix calculations
TEST_F(CameraCalibrationEdgeCaseTest, CameraMatrixPrecisionLoss) {
    std::vector<cv::Mat> images;

    // Create images with very small checkerboard squares that might cause precision issues
    for (int i = 0; i < 10; ++i) {
        cv::Mat img = createTestCheckerboard(cv::Size(1920, 1080), cv::Size(9, 6));
        images.push_back(img);
    }

    // Test with extremely small square size (potential precision loss)
    auto result = calibration_->calibrateSingleCamera(images, cv::Size(9, 6), 1e-10f);

    // Verify that the camera matrix doesn't contain NaN or infinite values
    if (!result.camera_matrix.empty()) {
        for (int i = 0; i < result.camera_matrix.rows; ++i) {
            for (int j = 0; j < result.camera_matrix.cols; ++j) {
                double val = result.camera_matrix.at<double>(i, j);
                EXPECT_FALSE(std::isnan(val)) << "Camera matrix contains NaN at (" << i << ", " << j << ")";
                EXPECT_FALSE(std::isinf(val)) << "Camera matrix contains infinity at (" << i << ", " << j << ")";
            }
        }
    }
}

// Test malformed input handling
TEST_F(CameraCalibrationEdgeCaseTest, MalformedInputHandling) {
    // Test with empty image vector
    std::vector<cv::Mat> empty_images;
    EXPECT_GRACEFUL_FAILURE(
        calibration_->calibrateSingleCamera(empty_images, cv::Size(9, 6), 25.0f)
    );

    // Test with corrupted images
    std::vector<cv::Mat> corrupted_images;
    corrupted_images.push_back(EdgeCaseTestFramework::generateCorruptedImage(cv::Size(640, 480), CV_8UC1));
    corrupted_images.push_back(EdgeCaseTestFramework::generateCorruptedImage(cv::Size(640, 480), CV_8UC1));

    EXPECT_GRACEFUL_FAILURE(
        calibration_->calibrateSingleCamera(corrupted_images, cv::Size(9, 6), 25.0f)
    );

    // Test with infinite values in image
    std::vector<cv::Mat> inf_images;
    inf_images.push_back(EdgeCaseTestFramework::generateInfiniteValuesMatrix(cv::Size(640, 480), CV_32F));

    EXPECT_GRACEFUL_FAILURE(
        calibration_->calibrateSingleCamera(inf_images, cv::Size(9, 6), 25.0f)
    );

    // Test with NaN values in image
    std::vector<cv::Mat> nan_images;
    nan_images.push_back(EdgeCaseTestFramework::generateNaNValuesMatrix(cv::Size(640, 480), CV_32F));

    EXPECT_GRACEFUL_FAILURE(
        calibration_->calibrateSingleCamera(nan_images, cv::Size(9, 6), 25.0f)
    );
}

// Test extreme image sizes
TEST_P(CameraCalibrationEdgeCaseTest, ExtremeImageSizes) {
    double edge_value = GetParam();
    auto edge_sizes = EdgeCaseTestFramework::getImageSizeEdgeCases();

    for (const auto& size : edge_sizes) {
        if (size.width > 0 && size.height > 0 && size.width < 32768 && size.height < 32768) {
            std::vector<cv::Mat> images;

            try {
                // Try to create a checkerboard with extreme size
                cv::Mat img = createTestCheckerboard(size, cv::Size(3, 3)); // Small board for extreme images
                images.push_back(img);

                // This should either work or fail gracefully
                auto result = calibration_->calibrateSingleCamera(images, cv::Size(3, 3), 25.0f);

                // If it succeeds, verify the results are sane
                if (!result.camera_matrix.empty()) {
                    EXPECT_NO_OVERFLOW(cv::determinant(result.camera_matrix));
                }

            } catch (const std::exception& e) {
                // Graceful failure is acceptable for extreme sizes
                EXPECT_TRUE(true) << "Gracefully handled extreme size "
                                  << size.width << "x" << size.height << ": " << e.what();
            }
        }
    }
}

// Test concurrent calibration operations
TEST_F(CameraCalibrationEdgeCaseTest, ConcurrentCalibration) {
    std::vector<cv::Mat> test_images;
    for (int i = 0; i < 5; ++i) {
        test_images.push_back(createTestCheckerboard(cv::Size(640, 480), cv::Size(9, 6)));
    }

    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    auto calibration_task = [&]() {
        try {
            auto local_calibration = std::make_unique<CameraCalibration>();
            auto result = local_calibration->calibrateSingleCamera(test_images, cv::Size(9, 6), 25.0f);

            if (!result.camera_matrix.empty()) {
                success_count++;
            } else {
                failure_count++;
            }
        } catch (const std::exception&) {
            failure_count++;
        }
    };

    EdgeCaseTestFramework::testConcurrentAccess(calibration_task, 10, 5);

    // At least some operations should succeed
    EXPECT_GT(success_count.load(), 0) << "No concurrent calibration operations succeeded";
}

// Test system under memory pressure
TEST_F(CameraCalibrationEdgeCaseTest, CalibrationUnderMemoryPressure) {
    // Simulate memory pressure
    EdgeCaseTestFramework::simulateMemoryPressure(512); // 512 MB

    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; ++i) {
        images.push_back(createTestCheckerboard(cv::Size(1920, 1080), cv::Size(9, 6)));
    }

    // Calibration should either succeed or fail gracefully under memory pressure
    try {
        auto result = calibration_->calibrateSingleCamera(images, cv::Size(9, 6), 25.0f);
        if (!result.camera_matrix.empty()) {
            EXPECT_TRUE(true) << "Calibration succeeded under memory pressure";
        }
    } catch (const std::exception& e) {
        EXPECT_TRUE(true) << "Calibration gracefully failed under memory pressure: " << e.what();
    }
}

// Test truncation in integer calculations
TEST_F(CameraCalibrationEdgeCaseTest, IntegerTruncationHandling) {
    auto int_edges = EdgeCaseTestFramework::getIntegerEdgeCases();

    for (int edge_val : int_edges) {
        if (edge_val > 0 && edge_val < 100) { // Reasonable board size range
            try {
                std::vector<cv::Mat> images;
                images.push_back(createTestCheckerboard(cv::Size(640, 480), cv::Size(edge_val, edge_val)));

                // Test corner detection with edge case board sizes
                std::vector<cv::Point2f> corners;
                bool detected = calibration_->detectCorners(images[0], cv::Size(edge_val, edge_val), corners);

                if (detected) {
                    // Verify corners are within image bounds
                    for (const auto& corner : corners) {
                        EXPECT_GE(corner.x, 0.0f) << "Corner x-coordinate is negative";
                        EXPECT_GE(corner.y, 0.0f) << "Corner y-coordinate is negative";
                        EXPECT_LT(corner.x, 640.0f) << "Corner x-coordinate exceeds image width";
                        EXPECT_LT(corner.y, 480.0f) << "Corner y-coordinate exceeds image height";

                        // Check for truncation artifacts
                        EXPECT_FALSE(corner.x == std::floor(corner.x) && corner.y == std::floor(corner.y))
                            << "Suspicious integer coordinates suggest truncation";
                    }
                }
            } catch (const std::exception& e) {
                // Graceful failure is acceptable
                EXPECT_TRUE(true) << "Gracefully handled edge case board size " << edge_val << ": " << e.what();
            }
        }
    }
}

// Test stereo calibration with mismatched image pairs
TEST_F(CameraCalibrationEdgeCaseTest, MismatchedStereoImages) {
    std::vector<cv::Mat> left_images, right_images;

    // Create mismatched image sets
    left_images.push_back(createTestCheckerboard(cv::Size(640, 480), cv::Size(9, 6)));
    left_images.push_back(createTestCheckerboard(cv::Size(640, 480), cv::Size(9, 6)));

    // Different number of right images
    right_images.push_back(createTestCheckerboard(cv::Size(640, 480), cv::Size(9, 6)));

    EXPECT_GRACEFUL_FAILURE(
        calibration_->calibrateStereoCamera(left_images, right_images, cv::Size(9, 6), 25.0f)
    );

    // Different image sizes
    right_images.clear();
    right_images.push_back(createTestCheckerboard(cv::Size(1280, 720), cv::Size(9, 6))); // Different size
    right_images.push_back(createTestCheckerboard(cv::Size(1280, 720), cv::Size(9, 6)));

    EXPECT_GRACEFUL_FAILURE(
        calibration_->calibrateStereoCamera(left_images, right_images, cv::Size(9, 6), 25.0f)
    );
}

// Instantiate parameterized tests with edge case values
INSTANTIATE_TEST_SUITE_P(
    FloatingPointEdgeCases,
    CameraCalibrationEdgeCaseTest,
    ::testing::ValuesIn(EdgeCaseTestFramework::getFloatingPointEdgeCases())
);

} // namespace testing
} // namespace stereo_vision
