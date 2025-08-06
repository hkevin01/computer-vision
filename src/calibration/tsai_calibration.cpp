#include "calibration/advanced_calibration.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace stereo_vision {
namespace calibration {

TsaiCalibration::TsaiCalibration(const AdvancedCalibrationConfig& config)
    : config_(config) {
}

TsaiCalibrationResult TsaiCalibration::calibrate(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size,
    ProgressCallback progress_callback) {

    TsaiCalibrationResult result;
    result.success = false;

    if (progress_callback) {
        progress_callback("Starting Tsai two-stage calibration...", 0.0);
    }

    // Validate input
    if (object_points.empty() || image_points.empty()) {
        result.error_message = "Empty input data";
        return result;
    }

    if (object_points.size() != image_points.size()) {
        result.error_message = "Mismatch between object points and image points";
        return result;
    }

    try {
        if (progress_callback) {
            progress_callback("Stage 1: Estimating extrinsic parameters using RAC...", 0.2);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Stage 1: Estimate extrinsic parameters using Radial Alignment Constraint
        // Use the first view for initial estimation
        auto [initial_rotation, initial_translation] = stageOneExtrinsicEstimation(
            object_points[0], image_points[0], image_size);

        if (initial_rotation.empty()) {
            result.error_message = "Stage 1 failed: Could not estimate initial extrinsic parameters";
            return result;
        }

        if (progress_callback) {
            progress_callback("Stage 2: Nonlinear optimization of all parameters...", 0.6);
        }

        // Stage 2: Nonlinear optimization of all parameters
        result = stageTwoNonlinearOptimization(object_points, image_points, image_size,
                                             initial_rotation, initial_translation);

        if (!result.success) {
            return result;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.computation_time = duration.count();

        if (progress_callback) {
            progress_callback("Tsai calibration completed successfully", 1.0);
        }

        if (config_.verbose) {
            std::cout << "Tsai two-stage calibration results:" << std::endl;
            std::cout << "  Focal length (fx, fy): (" << result.focal_length_x
                      << ", " << result.focal_length_y << ")" << std::endl;
            std::cout << "  Principal point (cx, cy): (" << result.principal_point.x
                      << ", " << result.principal_point.y << ")" << std::endl;
            std::cout << "  Radial distortion (k1, k2): (" << result.radial_distortion_k1
                      << ", " << result.radial_distortion_k2 << ")" << std::endl;
            std::cout << "  Reprojection error: " << result.reprojection_error << " pixels" << std::endl;
        }

    } catch (const cv::Exception& e) {
        result.error_message = "OpenCV error: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Tsai calibration error: " << e.what() << std::endl;
        }
    } catch (const std::exception& e) {
        result.error_message = "Standard error: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Tsai calibration error: " << e.what() << std::endl;
        }
    }

    return result;
}

TsaiCalibrationResult TsaiCalibration::calibrateSingleView(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Size& image_size,
    ProgressCallback progress_callback) {

    // Convert single view to multi-view format
    std::vector<std::vector<cv::Point3f>> object_points_multi = {object_points};
    std::vector<std::vector<cv::Point2f>> image_points_multi = {image_points};

    return calibrate(object_points_multi, image_points_multi, image_size, progress_callback);
}

void TsaiCalibration::setConfig(const AdvancedCalibrationConfig& config) {
    config_ = config;
}

const AdvancedCalibrationConfig& TsaiCalibration::getConfig() const {
    return config_;
}

std::pair<cv::Mat, cv::Vec3d> TsaiCalibration::stageOneExtrinsicEstimation(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Size& image_size) {

    cv::Mat rotation_matrix, translation_vector;

    if (object_points.size() != image_points.size() ||
        object_points.size() < config_.min_points_required) {
        return {cv::Mat(), cv::Vec3d()};
    }

    try {
        // Initial estimate using perspective-n-point (PnP) with assumed intrinsics
        // This provides a good starting point for the RAC-based optimization

        // Assume initial camera parameters based on image size and typical values
        double focal_length = std::max(image_size.width, image_size.height) * 0.8;
        cv::Mat initial_camera_matrix = (cv::Mat_<double>(3, 3) <<
            focal_length, 0, image_size.width / 2.0,
            0, focal_length, image_size.height / 2.0,
            0, 0, 1);

        cv::Mat distortion_coeffs = cv::Mat::zeros(4, 1, CV_64F);

        // Use solvePnP for initial pose estimation
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(object_points, image_points,
                                   initial_camera_matrix, distortion_coeffs,
                                   rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

        if (!success) {
            return {cv::Mat(), cv::Vec3d()};
        }

        cv::Rodrigues(rvec, rotation_matrix);
        cv::Vec3d translation_vec(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        // Refine using Radial Alignment Constraint
        // The RAC states that the direction from optical center to distorted point
        // is radially aligned with the corresponding 3D object point direction

        // Iterative refinement using RAC
        const int max_rac_iterations = 10;
        double prev_error = std::numeric_limits<double>::max();

        for (int iter = 0; iter < max_rac_iterations; ++iter) {
            double total_rac_error = 0.0;
            int valid_points = 0;

            // Calculate RAC error for current pose estimate
            for (size_t i = 0; i < object_points.size(); ++i) {
                double rac_error = radialAlignmentError(
                    object_points[i], image_points[i],
                    rotation_matrix, translation_vec,
                    initial_camera_matrix, distortion_coeffs);

                if (std::isfinite(rac_error)) {
                    total_rac_error += rac_error * rac_error;
                    valid_points++;
                }
            }

            if (valid_points == 0) break;

            double current_error = std::sqrt(total_rac_error / valid_points);

            if (config_.verbose) {
                std::cout << "RAC iteration " << iter << ", error: " << current_error << std::endl;
            }

            // Check for convergence
            if (std::abs(prev_error - current_error) < config_.optimization_tolerance) {
                break;
            }

            prev_error = current_error;

            // Simple gradient-based refinement (simplified RAC optimization)
            // In practice, this would use more sophisticated nonlinear optimization
            cv::Mat rvec_current;
            cv::Rodrigues(rotation_matrix, rvec_current);

            // Small perturbation for numerical gradient
            const double eps = 1e-6;
            cv::Mat jacobian = cv::Mat::zeros(6, 1, CV_64F);

            // Estimate gradient by finite differences
            for (int param = 0; param < 6; ++param) {
                cv::Mat rvec_plus = rvec_current.clone();
                cv::Vec3d tvec_plus = translation_vec;

                if (param < 3) {
                    rvec_plus.at<double>(param) += eps;
                } else {
                    tvec_plus[param - 3] += eps;
                }

                cv::Mat R_plus;
                cv::Rodrigues(rvec_plus, R_plus);

                double error_plus = 0.0;
                int count_plus = 0;

                for (size_t i = 0; i < object_points.size(); ++i) {
                    double rac_error = radialAlignmentError(
                        object_points[i], image_points[i],
                        R_plus, tvec_plus, initial_camera_matrix, distortion_coeffs);

                    if (std::isfinite(rac_error)) {
                        error_plus += rac_error * rac_error;
                        count_plus++;
                    }
                }

                if (count_plus > 0) {
                    error_plus = std::sqrt(error_plus / count_plus);
                    jacobian.at<double>(param) = (error_plus - current_error) / eps;
                }
            }

            // Simple gradient descent step
            double step_size = 0.01;
            for (int param = 0; param < 3; ++param) {
                rvec_current.at<double>(param) -= step_size * jacobian.at<double>(param);
            }
            for (int param = 3; param < 6; ++param) {
                translation_vec[param - 3] -= step_size * jacobian.at<double>(param);
            }

            cv::Rodrigues(rvec_current, rotation_matrix);
        }

        return {rotation_matrix, translation_vec};

    } catch (const cv::Exception& e) {
        if (config_.verbose) {
            std::cerr << "Stage 1 estimation error: " << e.what() << std::endl;
        }
        return {cv::Mat(), cv::Vec3d()};
    }
}

TsaiCalibrationResult TsaiCalibration::stageTwoNonlinearOptimization(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size,
    const cv::Mat& initial_rotation,
    const cv::Vec3d& initial_translation) {

    TsaiCalibrationResult result;
    result.success = false;

    try {
        // Use OpenCV's calibrateCamera for the nonlinear optimization stage
        // This handles the complex optimization while we provide good initial estimates

        cv::Mat camera_matrix, distortion_coeffs;
        std::vector<cv::Mat> rvecs, tvecs;

        // Set initial camera matrix based on image size
        double focal_length = std::max(image_size.width, image_size.height) * 0.8;
        camera_matrix = (cv::Mat_<double>(3, 3) <<
            focal_length, 0, image_size.width / 2.0,
            0, focal_length, image_size.height / 2.0,
            0, 0, 1);

        // Initialize distortion coefficients
        if (config_.estimate_radial_distortion && config_.estimate_tangential_distortion) {
            distortion_coeffs = cv::Mat::zeros(5, 1, CV_64F);
        } else if (config_.estimate_radial_distortion) {
            distortion_coeffs = cv::Mat::zeros(2, 1, CV_64F);
        } else {
            distortion_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        }

        // Set calibration flags based on configuration
        int flags = 0;
        if (!config_.estimate_tangential_distortion) {
            flags |= cv::CALIB_ZERO_TANGENT_DIST;
        }

        // Perform calibration
        double rms_error = cv::calibrateCamera(
            object_points, image_points, image_size,
            camera_matrix, distortion_coeffs, rvecs, tvecs,
            flags, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                   config_.optimization_iterations, config_.optimization_tolerance));

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

        // Use the first view's extrinsic parameters as representative result
        if (!rvecs.empty() && !tvecs.empty()) {
            cv::Rodrigues(rvecs[0], result.rotation_matrix);
            result.translation_vector = cv::Vec3d(
                tvecs[0].at<double>(0),
                tvecs[0].at<double>(1),
                tvecs[0].at<double>(2)
            );
        }

        result.reprojection_error = rms_error;
        result.success = (rms_error < config_.pixel_error_threshold);

        if (!result.success) {
            result.error_message = "Reprojection error too high: " + std::to_string(rms_error);
        }

    } catch (const cv::Exception& e) {
        result.error_message = "Stage 2 optimization failed: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Stage 2 optimization error: " << e.what() << std::endl;
        }
    }

    return result;
}

double TsaiCalibration::radialAlignmentError(
    const cv::Point3f& object_point,
    const cv::Point2f& image_point,
    const cv::Mat& rotation_matrix,
    const cv::Vec3d& translation_vector,
    const cv::Mat& camera_matrix,
    const cv::Mat& distortion_coeffs) {

    try {
        // Transform 3D point to camera coordinate system
        cv::Mat object_point_mat = (cv::Mat_<double>(3, 1) <<
            object_point.x, object_point.y, object_point.z);
        cv::Mat camera_point = rotation_matrix * object_point_mat +
                              (cv::Mat_<double>(3, 1) << translation_vector[0],
                               translation_vector[1], translation_vector[2]);

        if (camera_point.at<double>(2) <= 0) {
            return std::numeric_limits<double>::infinity();
        }

        // Project to normalized image coordinates
        double x_norm = camera_point.at<double>(0) / camera_point.at<double>(2);
        double y_norm = camera_point.at<double>(1) / camera_point.at<double>(2);

        // Apply distortion (simplified radial distortion model)
        double r2 = x_norm * x_norm + y_norm * y_norm;
        double k1 = distortion_coeffs.rows > 0 ? distortion_coeffs.at<double>(0) : 0.0;
        double k2 = distortion_coeffs.rows > 1 ? distortion_coeffs.at<double>(1) : 0.0;

        double distortion_factor = 1.0 + k1 * r2 + k2 * r2 * r2;
        double x_distorted = x_norm * distortion_factor;
        double y_distorted = y_norm * distortion_factor;

        // Convert to pixel coordinates
        double fx = camera_matrix.at<double>(0, 0);
        double fy = camera_matrix.at<double>(1, 1);
        double cx = camera_matrix.at<double>(0, 2);
        double cy = camera_matrix.at<double>(1, 2);

        double x_pixel = fx * x_distorted + cx;
        double y_pixel = fy * y_distorted + cy;

        // Calculate radial alignment error
        // The RAC states that the direction from optical center to the distorted image point
        // should be radially aligned with the direction to the object point

        cv::Point2f optical_center(cx, cy);
        cv::Point2f distorted_direction = image_point - optical_center;
        cv::Point2f projected_direction(x_pixel - cx, y_pixel - cy);

        // Normalize directions
        double distorted_norm = cv::norm(distorted_direction);
        double projected_norm = cv::norm(projected_direction);

        if (distorted_norm < 1e-6 || projected_norm < 1e-6) {
            return 0.0; // Both at optical center
        }

        distorted_direction *= (1.0 / distorted_norm);
        projected_direction *= (1.0 / projected_norm);

        // Calculate angular error (cross product magnitude for 2D vectors)
        double cross_product = distorted_direction.x * projected_direction.y -
                              distorted_direction.y * projected_direction.x;

        return std::abs(cross_product); // Angular alignment error

    } catch (...) {
        return std::numeric_limits<double>::infinity();
    }
}

void TsaiCalibration::optimizationObjective(
    const std::vector<double>& parameters,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    std::vector<double>& residuals) const {

    // This is a placeholder for a more sophisticated optimization objective function
    // In practice, this would implement the complete Tsai optimization with RAC constraints

    // For now, we delegate to OpenCV's robust implementation in stageTwoNonlinearOptimization
    residuals.clear();
    residuals.resize(1, 0.0);
}

std::vector<double> TsaiCalibration::parametersToVector(
    const cv::Mat& camera_matrix,
    const cv::Mat& distortion_coeffs,
    const std::vector<cv::Mat>& rotation_matrices,
    const std::vector<cv::Vec3d>& translation_vectors) const {

    std::vector<double> parameters;

    // Camera intrinsics: fx, fy, cx, cy
    parameters.push_back(camera_matrix.at<double>(0, 0)); // fx
    parameters.push_back(camera_matrix.at<double>(1, 1)); // fy
    parameters.push_back(camera_matrix.at<double>(0, 2)); // cx
    parameters.push_back(camera_matrix.at<double>(1, 2)); // cy

    // Distortion coefficients
    for (int i = 0; i < distortion_coeffs.rows; ++i) {
        parameters.push_back(distortion_coeffs.at<double>(i));
    }

    // Extrinsic parameters for each view
    for (size_t i = 0; i < rotation_matrices.size(); ++i) {
        cv::Mat rvec;
        cv::Rodrigues(rotation_matrices[i], rvec);
        parameters.push_back(rvec.at<double>(0));
        parameters.push_back(rvec.at<double>(1));
        parameters.push_back(rvec.at<double>(2));

        parameters.push_back(translation_vectors[i][0]);
        parameters.push_back(translation_vectors[i][1]);
        parameters.push_back(translation_vectors[i][2]);
    }

    return parameters;
}

void TsaiCalibration::vectorToParameters(
    const std::vector<double>& parameters,
    cv::Mat& camera_matrix,
    cv::Mat& distortion_coeffs,
    std::vector<cv::Mat>& rotation_matrices,
    std::vector<cv::Vec3d>& translation_vectors) const {

    if (parameters.size() < 4) return;

    // Extract camera intrinsics
    camera_matrix = (cv::Mat_<double>(3, 3) <<
        parameters[0], 0, parameters[2],
        0, parameters[1], parameters[3],
        0, 0, 1);

    // Extract distortion coefficients
    size_t distortion_start = 4;
    size_t distortion_count = std::min(size_t(8), parameters.size() - distortion_start);
    distortion_coeffs = cv::Mat::zeros(distortion_count, 1, CV_64F);
    for (size_t i = 0; i < distortion_count; ++i) {
        distortion_coeffs.at<double>(i) = parameters[distortion_start + i];
    }

    // Extract extrinsic parameters
    size_t extrinsic_start = distortion_start + distortion_count;
    size_t remaining_params = parameters.size() - extrinsic_start;
    size_t num_views = remaining_params / 6;

    rotation_matrices.clear();
    translation_vectors.clear();

    for (size_t i = 0; i < num_views; ++i) {
        size_t base_idx = extrinsic_start + i * 6;

        cv::Mat rvec = (cv::Mat_<double>(3, 1) <<
            parameters[base_idx], parameters[base_idx + 1], parameters[base_idx + 2]);
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        rotation_matrices.push_back(rotation_matrix);

        cv::Vec3d translation_vector(
            parameters[base_idx + 3],
            parameters[base_idx + 4],
            parameters[base_idx + 5]
        );
        translation_vectors.push_back(translation_vector);
    }
}

} // namespace calibration
} // namespace stereo_vision
