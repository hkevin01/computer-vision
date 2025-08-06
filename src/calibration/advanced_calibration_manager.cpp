#include "calibration/advanced_calibration.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

namespace stereo_vision {
namespace calibration {

AdvancedCalibrationManager::AdvancedCalibrationManager(const AdvancedCalibrationConfig& config)
    : config_(config) {
    eight_point_algorithm_ = std::make_unique<EightPointAlgorithm>(config);
    tsai_calibration_ = std::make_unique<TsaiCalibration>(config);
}

AdvancedCalibrationManager::~AdvancedCalibrationManager() = default;

CalibrationComparison AdvancedCalibrationManager::compareAlgorithms(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size,
    ProgressCallback progress_callback) {

    CalibrationComparison comparison;
    comparison.success = false;

    if (progress_callback) {
        progress_callback("Starting algorithm comparison...", 0.0);
    }

    // Validate input
    if (object_points.empty() || image_points.empty() ||
        object_points.size() != image_points.size()) {
        comparison.error_message = "Invalid input data for comparison";
        return comparison;
    }

    try {
        // Test OpenCV's standard calibration
        if (progress_callback) {
            progress_callback("Running OpenCV standard calibration...", 0.1);
        }
        comparison.opencv_result = runOpenCVCalibration(object_points, image_points, image_size);

        // Test Tsai's two-stage calibration
        if (progress_callback) {
            progress_callback("Running Tsai two-stage calibration...", 0.4);
        }
        comparison.tsai_result = tsai_calibration_->calibrate(
            object_points, image_points, image_size,
            [progress_callback](const std::string& msg, double sub_progress) {
                if (progress_callback) {
                    progress_callback("Tsai: " + msg, 0.4 + 0.3 * sub_progress);
                }
            });

        // Test Eight-Point Algorithm if we have stereo data
        if (object_points.size() >= 2 && progress_callback) {
            progress_callback("Running Eight-Point Algorithm...", 0.7);
        }

        if (object_points.size() >= 2) {
            // Convert first two views for eight-point algorithm
            comparison.eight_point_result = eight_point_algorithm_->estimatePose(
                image_points[0], image_points[1],
                [progress_callback](const std::string& msg, double sub_progress) {
                    if (progress_callback) {
                        progress_callback("Eight-Point: " + msg, 0.7 + 0.2 * sub_progress);
                    }
                });
        }

        if (progress_callback) {
            progress_callback("Analyzing results and generating comparison...", 0.9);
        }

        // Analyze and compare results
        comparison.analysis = analyzeResults(comparison);
        comparison.recommendation = generateRecommendation(comparison);
        comparison.success = true;

        if (progress_callback) {
            progress_callback("Algorithm comparison completed", 1.0);
        }

        if (config_.verbose) {
            printComparisonSummary(comparison);
        }

    } catch (const cv::Exception& e) {
        comparison.error_message = "OpenCV error during comparison: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Comparison error: " << e.what() << std::endl;
        }
    } catch (const std::exception& e) {
        comparison.error_message = "Standard error during comparison: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Comparison error: " << e.what() << std::endl;
        }
    }

    return comparison;
}

TsaiCalibrationResult AdvancedCalibrationManager::runOpenCVCalibration(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size) {

    TsaiCalibrationResult result;
    result.success = false;

    try {
        cv::Mat camera_matrix, distortion_coeffs;
        std::vector<cv::Mat> rvecs, tvecs;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Set calibration flags
        int flags = 0;
        if (!config_.estimate_tangential_distortion) {
            flags |= cv::CALIB_ZERO_TANGENT_DIST;
        }

        double rms_error = cv::calibrateCamera(
            object_points, image_points, image_size,
            camera_matrix, distortion_coeffs, rvecs, tvecs,
            flags, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                   config_.optimization_iterations, config_.optimization_tolerance));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Extract results
        result.camera_matrix = camera_matrix;
        result.distortion_coeffs = distortion_coeffs;
        result.focal_length_x = camera_matrix.at<double>(0, 0);
        result.focal_length_y = camera_matrix.at<double>(1, 1);
        result.principal_point.x = camera_matrix.at<double>(0, 2);
        result.principal_point.y = camera_matrix.at<double>(1, 2);

        // Extract distortion coefficients
        if (distortion_coeffs.rows >= 1) result.radial_distortion_k1 = distortion_coeffs.at<double>(0);
        if (distortion_coeffs.rows >= 2) result.radial_distortion_k2 = distortion_coeffs.at<double>(1);
        if (distortion_coeffs.rows >= 3) result.tangential_p1 = distortion_coeffs.at<double>(2);
        if (distortion_coeffs.rows >= 4) result.tangential_p2 = distortion_coeffs.at<double>(3);

        // Use first view's extrinsic parameters
        if (!rvecs.empty() && !tvecs.empty()) {
            cv::Rodrigues(rvecs[0], result.rotation_matrix);
            result.translation_vector = cv::Vec3d(
                tvecs[0].at<double>(0),
                tvecs[0].at<double>(1),
                tvecs[0].at<double>(2)
            );
        }

        result.reprojection_error = rms_error;
        result.computation_time = duration.count();
        result.success = (rms_error < config_.pixel_error_threshold);

        if (!result.success) {
            result.error_message = "OpenCV calibration: Reprojection error too high";
        }

    } catch (const cv::Exception& e) {
        result.error_message = "OpenCV calibration failed: " + std::string(e.what());
    }

    return result;
}

CalibrationAnalysis AdvancedCalibrationManager::analyzeResults(const CalibrationComparison& comparison) {
    CalibrationAnalysis analysis;

    // Collect successful results
    std::vector<std::pair<std::string, TsaiCalibrationResult>> successful_results;

    if (comparison.opencv_result.success) {
        successful_results.push_back({"OpenCV", comparison.opencv_result});
    }
    if (comparison.tsai_result.success) {
        successful_results.push_back({"Tsai", comparison.tsai_result});
    }

    if (successful_results.empty()) {
        analysis.summary = "No calibration algorithms succeeded";
        return analysis;
    }

    // Find best result by reprojection error
    auto best_result = std::min_element(successful_results.begin(), successful_results.end(),
        [](const auto& a, const auto& b) {
            return a.second.reprojection_error < b.second.reprojection_error;
        });

    analysis.best_algorithm = best_result->first;
    analysis.best_reprojection_error = best_result->second.reprojection_error;

    // Calculate statistics
    std::vector<double> errors;
    std::vector<double> computation_times;

    for (const auto& result : successful_results) {
        errors.push_back(result.second.reprojection_error);
        computation_times.push_back(result.second.computation_time);
    }

    analysis.mean_reprojection_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    analysis.std_reprojection_error = calculateStandardDeviation(errors, analysis.mean_reprojection_error);

    analysis.mean_computation_time = std::accumulate(computation_times.begin(), computation_times.end(), 0.0) / computation_times.size();
    analysis.std_computation_time = calculateStandardDeviation(computation_times, analysis.mean_computation_time);

    // Analyze algorithm characteristics
    if (comparison.opencv_result.success && comparison.tsai_result.success) {
        double opencv_error = comparison.opencv_result.reprojection_error;
        double tsai_error = comparison.tsai_result.reprojection_error;
        double error_difference = std::abs(opencv_error - tsai_error);

        if (error_difference < 0.1) {
            analysis.consistency = "High - algorithms agree within 0.1 pixels";
        } else if (error_difference < 0.5) {
            analysis.consistency = "Medium - algorithms agree within 0.5 pixels";
        } else {
            analysis.consistency = "Low - significant disagreement between algorithms";
        }

        // Check parameter consistency
        double focal_diff_x = std::abs(comparison.opencv_result.focal_length_x - comparison.tsai_result.focal_length_x);
        double focal_diff_y = std::abs(comparison.opencv_result.focal_length_y - comparison.tsai_result.focal_length_y);

        analysis.parameter_consistency = (focal_diff_x < 10.0 && focal_diff_y < 10.0) ?
            "Good" : "Poor";
    } else {
        analysis.consistency = "Cannot assess - insufficient successful calibrations";
        analysis.parameter_consistency = "Unknown";
    }

    // Generate summary
    std::ostringstream summary;
    summary << "Calibration Analysis Summary:\n";
    summary << "- " << successful_results.size() << " of " <<
               (comparison.opencv_result.success || !comparison.opencv_result.error_message.empty() ? 1 : 0) +
               (comparison.tsai_result.success || !comparison.tsai_result.error_message.empty() ? 1 : 0) +
               (comparison.eight_point_result.success || !comparison.eight_point_result.error_message.empty() ? 1 : 0)
               << " algorithms succeeded\n";
    summary << "- Best algorithm: " << analysis.best_algorithm
            << " (error: " << std::fixed << std::setprecision(3) << analysis.best_reprojection_error << " pixels)\n";
    summary << "- Algorithm consistency: " << analysis.consistency << "\n";
    summary << "- Parameter consistency: " << analysis.parameter_consistency;

    analysis.summary = summary.str();

    return analysis;
}

std::string AdvancedCalibrationManager::generateRecommendation(const CalibrationComparison& comparison) {
    std::ostringstream recommendation;

    recommendation << "Calibration Algorithm Recommendation:\n\n";

    // Count successful algorithms
    int successful_count = 0;
    if (comparison.opencv_result.success) successful_count++;
    if (comparison.tsai_result.success) successful_count++;
    if (comparison.eight_point_result.success) successful_count++;

    if (successful_count == 0) {
        recommendation << "❌ No algorithms succeeded. Recommendations:\n";
        recommendation << "  • Check input data quality and completeness\n";
        recommendation << "  • Verify calibration pattern detection accuracy\n";
        recommendation << "  • Consider using more calibration images\n";
        recommendation << "  • Review camera setup and lighting conditions\n";
        return recommendation.str();
    }

    // Analyze best performing algorithm
    std::vector<std::pair<std::string, double>> results;
    if (comparison.opencv_result.success) {
        results.push_back({"OpenCV Standard", comparison.opencv_result.reprojection_error});
    }
    if (comparison.tsai_result.success) {
        results.push_back({"Tsai Two-Stage", comparison.tsai_result.reprojection_error});
    }

    if (!results.empty()) {
        auto best = std::min_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        recommendation << "🎯 Primary Recommendation: " << best->first << "\n";
        recommendation << "   Reprojection Error: " << std::fixed << std::setprecision(3)
                      << best->second << " pixels\n\n";

        // Specific algorithm recommendations
        if (best->first == "OpenCV Standard") {
            recommendation << "✅ OpenCV Standard Calibration is recommended because:\n";
            recommendation << "  • Proven robust implementation with extensive optimization\n";
            recommendation << "  • Good balance of accuracy and computational efficiency\n";
            recommendation << "  • Handles various distortion models effectively\n";
            recommendation << "  • Well-tested across diverse camera types\n\n";
        } else if (best->first == "Tsai Two-Stage") {
            recommendation << "✅ Tsai Two-Stage Calibration is recommended because:\n";
            recommendation << "  • Excellent for high-precision applications\n";
            recommendation << "  • RAC constraint provides superior distortion handling\n";
            recommendation << "  • Two-stage approach reduces local minima issues\n";
            recommendation << "  • Optimal for single-view or limited multi-view scenarios\n\n";
        }

        // Secondary recommendations
        recommendation << "💡 Additional Insights:\n";

        if (comparison.opencv_result.success && comparison.tsai_result.success) {
            double error_diff = std::abs(comparison.opencv_result.reprojection_error -
                                       comparison.tsai_result.reprojection_error);
            if (error_diff < 0.1) {
                recommendation << "  • Both algorithms show excellent agreement (< 0.1px difference)\n";
                recommendation << "  • Either algorithm can be used with confidence\n";
            } else if (error_diff < 0.5) {
                recommendation << "  • Algorithms show reasonable agreement (< 0.5px difference)\n";
                recommendation << "  • Consider application requirements for final choice\n";
            } else {
                recommendation << "  • Significant difference between algorithms detected\n";
                recommendation << "  • Recommend additional validation with test images\n";
            }
        }

        // Eight-point algorithm assessment
        if (comparison.eight_point_result.success) {
            recommendation << "  • Eight-Point Algorithm succeeded for stereo estimation\n";
            recommendation << "  • Consider for stereo vision applications\n";
        } else if (!comparison.eight_point_result.error_message.empty()) {
            recommendation << "  • Eight-Point Algorithm available for stereo scenarios\n";
            recommendation << "  • Requires paired stereo images for operation\n";
        }

        // Usage guidelines
        recommendation << "\n📋 Usage Guidelines:\n";

        if (best->second < 0.5) {
            recommendation << "  • Excellent calibration quality achieved\n";
            recommendation << "  • Suitable for high-precision applications\n";
        } else if (best->second < 1.0) {
            recommendation << "  • Good calibration quality achieved\n";
            recommendation << "  • Suitable for general computer vision tasks\n";
        } else {
            recommendation << "  • Moderate calibration quality\n";
            recommendation << "  • Consider improving data collection for better results\n";
        }

        recommendation << "  • Validate results with independent test images\n";
        recommendation << "  • Consider application-specific error requirements\n";
    }

    return recommendation.str();
}

void AdvancedCalibrationManager::printComparisonSummary(const CalibrationComparison& comparison) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ADVANCED CALIBRATION ALGORITHM COMPARISON" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // OpenCV Results
    std::cout << "\n🔧 OpenCV Standard Calibration:" << std::endl;
    if (comparison.opencv_result.success) {
        std::cout << "  ✅ Status: SUCCESS" << std::endl;
        std::cout << "  📏 Reprojection Error: " << std::fixed << std::setprecision(3)
                  << comparison.opencv_result.reprojection_error << " pixels" << std::endl;
        std::cout << "  ⏱️  Computation Time: " << comparison.opencv_result.computation_time << " ms" << std::endl;
        std::cout << "  🎯 Focal Length (fx, fy): ("
                  << std::setprecision(2) << comparison.opencv_result.focal_length_x << ", "
                  << comparison.opencv_result.focal_length_y << ")" << std::endl;
        std::cout << "  📍 Principal Point (cx, cy): ("
                  << std::setprecision(2) << comparison.opencv_result.principal_point.x << ", "
                  << comparison.opencv_result.principal_point.y << ")" << std::endl;
    } else {
        std::cout << "  ❌ Status: FAILED" << std::endl;
        std::cout << "  🚫 Error: " << comparison.opencv_result.error_message << std::endl;
    }

    // Tsai Results
    std::cout << "\n🔬 Tsai Two-Stage Calibration:" << std::endl;
    if (comparison.tsai_result.success) {
        std::cout << "  ✅ Status: SUCCESS" << std::endl;
        std::cout << "  📏 Reprojection Error: " << std::fixed << std::setprecision(3)
                  << comparison.tsai_result.reprojection_error << " pixels" << std::endl;
        std::cout << "  ⏱️  Computation Time: " << comparison.tsai_result.computation_time << " ms" << std::endl;
        std::cout << "  🎯 Focal Length (fx, fy): ("
                  << std::setprecision(2) << comparison.tsai_result.focal_length_x << ", "
                  << comparison.tsai_result.focal_length_y << ")" << std::endl;
        std::cout << "  📍 Principal Point (cx, cy): ("
                  << std::setprecision(2) << comparison.tsai_result.principal_point.x << ", "
                  << comparison.tsai_result.principal_point.y << ")" << std::endl;
    } else {
        std::cout << "  ❌ Status: FAILED" << std::endl;
        std::cout << "  🚫 Error: " << comparison.tsai_result.error_message << std::endl;
    }

    // Eight-Point Results
    std::cout << "\n⚡ Eight-Point Algorithm:" << std::endl;
    if (comparison.eight_point_result.success) {
        std::cout << "  ✅ Status: SUCCESS" << std::endl;
        std::cout << "  📏 Estimation Error: " << std::fixed << std::setprecision(3)
                  << comparison.eight_point_result.estimation_error << " pixels" << std::endl;
        std::cout << "  ⏱️  Computation Time: " << comparison.eight_point_result.computation_time << " ms" << std::endl;
        std::cout << "  🔄 Pose Available: " << (comparison.eight_point_result.rotation_matrix.empty() ? "No" : "Yes") << std::endl;
    } else {
        std::cout << "  ❌ Status: FAILED" << std::endl;
        std::cout << "  🚫 Error: " << comparison.eight_point_result.error_message << std::endl;
    }

    // Analysis Summary
    std::cout << "\n📊 Analysis Summary:" << std::endl;
    std::cout << comparison.analysis.summary << std::endl;

    // Recommendation
    std::cout << "\n" << comparison.recommendation << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

double AdvancedCalibrationManager::calculateStandardDeviation(
    const std::vector<double>& values, double mean) {

    if (values.size() <= 1) return 0.0;

    double sum_squared_diff = 0.0;
    for (double value : values) {
        double diff = value - mean;
        sum_squared_diff += diff * diff;
    }

    return std::sqrt(sum_squared_diff / (values.size() - 1));
}

void AdvancedCalibrationManager::setConfig(const AdvancedCalibrationConfig& config) {
    config_ = config;
    if (eight_point_algorithm_) {
        eight_point_algorithm_->setConfig(config);
    }
    if (tsai_calibration_) {
        tsai_calibration_->setConfig(config);
    }
}

const AdvancedCalibrationConfig& AdvancedCalibrationManager::getConfig() const {
    return config_;
}

} // namespace calibration
} // namespace stereo_vision
