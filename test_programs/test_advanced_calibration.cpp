#include "calibration/advanced_calibration.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace stereo_vision::calibration;

int main() {
    std::cout << "Testing Advanced Camera Calibration Module" << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        // Create calibration configuration
        AdvancedCalibrationConfig config;
        config.verbose = true;
        config.pixel_error_threshold = 1.0;

        // Create calibration manager
        AdvancedCalibrationManager manager(config);

        std::cout << "✅ Successfully created AdvancedCalibrationManager" << std::endl;

        // Create Eight-Point Algorithm instance
        EightPointAlgorithm eight_point(config);

        std::cout << "✅ Successfully created EightPointAlgorithm" << std::endl;

        // Create Tsai Calibration instance
        TsaiCalibration tsai(config);

        std::cout << "✅ Successfully created TsaiCalibration" << std::endl;

        // Test with synthetic data (simple planar pattern)
        std::vector<std::vector<cv::Point3f>> object_points;
        std::vector<std::vector<cv::Point2f>> image_points;

        // Create a simple 3D calibration pattern (chessboard-like)
        std::vector<cv::Point3f> pattern_3d;
        std::vector<cv::Point2f> pattern_2d;

        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 8; ++j) {
                // 3D pattern points (Z=0 plane)
                pattern_3d.push_back(cv::Point3f(j * 30.0f, i * 30.0f, 0.0f));

                // Simulated 2D image points (with simple projection)
                float fx = 800.0f, fy = 800.0f;
                float cx = 320.0f, cy = 240.0f;
                float x = (j * 30.0f * fx / 300.0f) + cx;
                float y = (i * 30.0f * fy / 300.0f) + cy;
                pattern_2d.push_back(cv::Point2f(x, y));
            }
        }

        object_points.push_back(pattern_3d);
        image_points.push_back(pattern_2d);

        cv::Size image_size(640, 480);

        std::cout << "📊 Testing with synthetic calibration data:" << std::endl;
        std::cout << "   - Pattern size: " << pattern_3d.size() << " points" << std::endl;
        std::cout << "   - Image size: " << image_size.width << "x" << image_size.height << std::endl;

        // Test Tsai calibration
        std::cout << "\n🔬 Testing Tsai Two-Stage Calibration..." << std::endl;
        auto tsai_result = tsai.calibrate(object_points, image_points, image_size,
            [](const std::string& msg, double progress) {
                std::cout << "   Progress: " << std::fixed << std::setprecision(1)
                          << (progress * 100.0) << "% - " << msg << std::endl;
            });

        if (tsai_result.success) {
            std::cout << "✅ Tsai calibration succeeded!" << std::endl;
            std::cout << "   Reprojection error: " << std::fixed << std::setprecision(3)
                      << tsai_result.reprojection_error << " pixels" << std::endl;
            std::cout << "   Computation time: " << tsai_result.computation_time << " ms" << std::endl;
        } else {
            std::cout << "❌ Tsai calibration failed: " << tsai_result.error_message << std::endl;
        }

        // Test Eight-Point Algorithm (requires at least two point sets)
        if (pattern_2d.size() >= 8) {
            std::cout << "\n⚡ Testing Eight-Point Algorithm..." << std::endl;

            // Create two slightly different point sets to simulate stereo
            std::vector<cv::Point2f> points1, points2;
            for (size_t i = 0; i < std::min(size_t(16), pattern_2d.size()); ++i) {
                points1.push_back(pattern_2d[i]);
                // Simulate stereo by adding small disparity
                points2.push_back(cv::Point2f(pattern_2d[i].x - 5.0f, pattern_2d[i].y));
            }

            auto eight_point_result = eight_point.estimatePose(points1, points2,
                [](const std::string& msg, double progress) {
                    std::cout << "   Progress: " << std::fixed << std::setprecision(1)
                              << (progress * 100.0) << "% - " << msg << std::endl;
                });

            if (eight_point_result.success) {
                std::cout << "✅ Eight-Point algorithm succeeded!" << std::endl;
                std::cout << "   Estimation error: " << std::fixed << std::setprecision(3)
                          << eight_point_result.estimation_error << " pixels" << std::endl;
                std::cout << "   Computation time: " << eight_point_result.computation_time << " ms" << std::endl;
                std::cout << "   Inliers: " << eight_point_result.inlier_count << "/" << points1.size() << std::endl;
            } else {
                std::cout << "❌ Eight-Point algorithm failed: " << eight_point_result.error_message << std::endl;
            }
        }

        // Test comparison functionality (requires more realistic data for meaningful comparison)
        std::cout << "\n🎯 Testing Algorithm Comparison..." << std::endl;
        auto comparison = manager.compareAlgorithms(object_points, image_points, image_size,
            [](const std::string& msg, double progress) {
                std::cout << "   Progress: " << std::fixed << std::setprecision(1)
                          << (progress * 100.0) << "% - " << msg << std::endl;
            });

        if (comparison.success) {
            std::cout << "✅ Algorithm comparison completed!" << std::endl;
            std::cout << "\n📋 Comparison Summary:" << std::endl;
            std::cout << comparison.analysis.summary << std::endl;
            std::cout << "\n💡 Recommendation:" << std::endl;
            std::cout << comparison.recommendation << std::endl;
        } else {
            std::cout << "❌ Algorithm comparison failed: " << comparison.error_message << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error during testing: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n🎉 Advanced Camera Calibration Module Test Complete!" << std::endl;
    return 0;
}
