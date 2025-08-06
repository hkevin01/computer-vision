#include "calibration/advanced_calibration.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace stereo_vision {
namespace calibration {

EightPointAlgorithm::EightPointAlgorithm(const AdvancedCalibrationConfig& config)
    : config_(config) {
}

EightPointResult EightPointAlgorithm::calibrate(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cv::Size& image_size,
    ProgressCallback progress_callback) {

    EightPointResult result;
    result.success = false;
    result.computation_time = 0.0;
    result.estimation_error = std::numeric_limits<double>::max();

    auto start_time = std::chrono::high_resolution_clock::now();

    if (progress_callback) {
        progress_callback("Starting eight-point algorithm calibration...", 0.0);
    }

    // Validate input
    if (points1.size() != points2.size()) {
        result.error_message = "Point sets must have equal size";
        return result;
    }

    if (points1.size() < config_.min_points_required) {
        result.error_message = "Insufficient points for calibration (minimum: " +
                              std::to_string(config_.min_points_required) + ")";
        return result;
    }

    if (progress_callback) {
        progress_callback("Estimating fundamental matrix...", 0.2);
    }

    try {
        // Step 1: Estimate fundamental matrix using eight-point algorithm
        std::vector<bool> inlier_mask;
        cv::Mat fundamental_matrix = estimateFundamentalMatrix(points1, points2, inlier_mask);

        if (fundamental_matrix.empty()) {
            result.error_message = "Failed to estimate fundamental matrix";
            return result;
        }

        result.fundamental_matrix = fundamental_matrix;
        result.inlier_mask = inlier_mask;
        result.inlier_count = std::count(inlier_mask.begin(), inlier_mask.end(), true);

        if (progress_callback) {
            progress_callback("Fundamental matrix estimated successfully", 0.5);
        }

        // For eight-point algorithm without known intrinsics, we estimate them
        // using basic assumptions about the camera
        cv::Mat camera_matrix1 = (cv::Mat_<double>(3, 3) <<
            image_size.width * 0.8, 0, image_size.width / 2.0,
            0, image_size.width * 0.8, image_size.height / 2.0,
            0, 0, 1);

        cv::Mat camera_matrix2 = camera_matrix1.clone();

        if (progress_callback) {
            progress_callback("Converting to essential matrix...", 0.6);
        }

        // Step 2: Convert fundamental matrix to essential matrix
        cv::Mat essential_matrix = fundamentalToEssential(fundamental_matrix,
                                                         camera_matrix1, camera_matrix2);
        result.essential_matrix = essential_matrix;

        if (progress_callback) {
            progress_callback("Decomposing essential matrix...", 0.7);
        }

        // Step 3: Decompose essential matrix to get possible poses
        auto possible_poses = decomposeEssentialMatrix(essential_matrix);

        if (progress_callback) {
            progress_callback("Selecting correct pose...", 0.8);
        }

        // Step 4: Select correct pose by testing positive depth constraint
        auto [rotation_matrix, translation_vector] = selectCorrectPose(
            possible_poses, points1, points2, camera_matrix1, camera_matrix2,
            result.triangulated_points);

        result.rotation_matrix = rotation_matrix;
        result.translation_vector = translation_vector;

        if (progress_callback) {
            progress_callback("Calculating reprojection error...", 0.9);
        }

        // Step 5: Calculate reprojection error
        result.reprojection_error = calculateReprojectionError(
            result.triangulated_points, points2, camera_matrix2,
            rotation_matrix, translation_vector);

        if (result.reprojection_error > config_.pixel_error_threshold) {
            result.error_message = "Reprojection error too high: " +
                                 std::to_string(result.reprojection_error);
            if (config_.verbose) {
                std::cout << "Warning: " << result.error_message << std::endl;
            }
        }

        result.success = true;
        result.estimation_error = result.reprojection_error;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.computation_time = duration.count();

        if (progress_callback) {
            progress_callback("Eight-point calibration completed successfully", 1.0);
        }

        if (config_.verbose) {
            std::cout << "Eight-point algorithm results:" << std::endl;
            std::cout << "  Inliers: " << result.inlier_count << "/" << points1.size() << std::endl;
            std::cout << "  Reprojection error: " << result.reprojection_error << " pixels" << std::endl;
            std::cout << "  Triangulated points: " << result.triangulated_points.size() << std::endl;
        }

    } catch (const cv::Exception& e) {
        result.error_message = "OpenCV error: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Eight-point algorithm error: " << e.what() << std::endl;
        }
    } catch (const std::exception& e) {
        result.error_message = "Standard error: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Eight-point algorithm error: " << e.what() << std::endl;
        }
    }

    return result;
}

EightPointResult EightPointAlgorithm::calibrateWithIntrinsics(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cv::Mat& camera_matrix1,
    const cv::Mat& camera_matrix2,
    ProgressCallback progress_callback) {

    EightPointResult result;
    result.success = false;
    result.computation_time = 0.0;
    result.estimation_error = std::numeric_limits<double>::max();

    auto start_time = std::chrono::high_resolution_clock::now();

    if (progress_callback) {
        progress_callback("Starting eight-point algorithm with known intrinsics...", 0.0);
    }

    // Validate input
    if (points1.size() != points2.size()) {
        result.error_message = "Point sets must have equal size";
        return result;
    }

    if (points1.size() < config_.min_points_required) {
        result.error_message = "Insufficient points for calibration";
        return result;
    }

    try {
        if (progress_callback) {
            progress_callback("Estimating essential matrix directly...", 0.2);
        }

        // Use OpenCV's findEssentialMat for robust estimation
        cv::Mat essential_matrix;
        std::vector<uchar> inlier_mask_uchar;

        if (config_.use_ransac) {
            essential_matrix = cv::findEssentialMat(
                points1, points2, camera_matrix1, cv::RANSAC,
                config_.ransac_confidence, config_.ransac_threshold, inlier_mask_uchar);
        } else {
            essential_matrix = cv::findEssentialMat(
                points1, points2, camera_matrix1, cv::LMEDS,
                config_.ransac_confidence, config_.ransac_threshold, inlier_mask_uchar);
        }

        result.essential_matrix = essential_matrix;

        // Convert uchar mask to bool
        result.inlier_mask.resize(inlier_mask_uchar.size());
        std::transform(inlier_mask_uchar.begin(), inlier_mask_uchar.end(),
                      result.inlier_mask.begin(), [](uchar x) { return x != 0; });

        result.inlier_count = std::count(result.inlier_mask.begin(), result.inlier_mask.end(), true);

        if (progress_callback) {
            progress_callback("Recovering pose from essential matrix...", 0.6);
        }

        // Recover pose from essential matrix
        cv::Mat rotation_matrix, translation_vector;
        int points_in_front = cv::recoverPose(essential_matrix, points1, points2,
                                             camera_matrix1, rotation_matrix, translation_vector,
                                             inlier_mask_uchar);

        result.rotation_matrix = rotation_matrix;
        result.translation_vector = cv::Vec3d(translation_vector.at<double>(0),
                                            translation_vector.at<double>(1),
                                            translation_vector.at<double>(2));

        if (progress_callback) {
            progress_callback("Triangulating 3D points...", 0.8);
        }

        // Triangulate 3D points
        cv::Mat projection_matrix1 = camera_matrix1 * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat rt_matrix = (cv::Mat_<double>(3, 4) <<
            rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1), rotation_matrix.at<double>(0, 2), translation_vector.at<double>(0),
            rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2), translation_vector.at<double>(1),
            rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2), translation_vector.at<double>(2));
        cv::Mat projection_matrix2 = camera_matrix2 * rt_matrix;

        cv::Mat points_4d;
        cv::triangulatePoints(projection_matrix1, projection_matrix2, points1, points2, points_4d);

        // Convert homogeneous coordinates to 3D points
        result.triangulated_points.clear();
        for (int i = 0; i < points_4d.cols; ++i) {
            if (result.inlier_mask[i]) {
                cv::Vec4f homogeneous_point = points_4d.col(i);
                if (std::abs(homogeneous_point[3]) > 1e-6) {
                    result.triangulated_points.emplace_back(
                        homogeneous_point[0] / homogeneous_point[3],
                        homogeneous_point[1] / homogeneous_point[3],
                        homogeneous_point[2] / homogeneous_point[3]
                    );
                }
            }
        }

        if (progress_callback) {
            progress_callback("Calculating reprojection error...", 0.9);
        }

        // Calculate reprojection error
        result.reprojection_error = calculateReprojectionError(
            result.triangulated_points, points2, camera_matrix2,
            rotation_matrix, result.translation_vector);

        // Convert essential matrix to fundamental matrix for completeness
        cv::Mat camera_matrix1_inv, camera_matrix2_inv_t;
        cv::invert(camera_matrix1, camera_matrix1_inv);
        cv::invert(camera_matrix2.t(), camera_matrix2_inv_t);
        result.fundamental_matrix = camera_matrix2_inv_t * essential_matrix * camera_matrix1_inv;

        result.success = true;
        result.estimation_error = result.reprojection_error;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.computation_time = duration.count();

        if (progress_callback) {
            progress_callback("Calibration completed successfully", 1.0);
        }

        if (config_.verbose) {
            std::cout << "Eight-point algorithm with intrinsics results:" << std::endl;
            std::cout << "  Inliers: " << result.inlier_count << "/" << points1.size() << std::endl;
            std::cout << "  Points in front: " << points_in_front << std::endl;
            std::cout << "  Reprojection error: " << result.reprojection_error << " pixels" << std::endl;
            std::cout << "  Triangulated points: " << result.triangulated_points.size() << std::endl;
        }

    } catch (const cv::Exception& e) {
        result.error_message = "OpenCV error: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Eight-point algorithm error: " << e.what() << std::endl;
        }
    } catch (const std::exception& e) {
        result.error_message = "Standard error: " + std::string(e.what());
        if (config_.verbose) {
            std::cerr << "Eight-point algorithm error: " << e.what() << std::endl;
        }
    }

    return result;
}

EightPointResult EightPointAlgorithm::estimatePose(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    ProgressCallback progress_callback) {

    // Use default image size for pose estimation
    cv::Size default_size(1000, 1000);
    return calibrate(points1, points2, default_size, progress_callback);
}

void EightPointAlgorithm::setConfig(const AdvancedCalibrationConfig& config) {
    config_ = config;
}

const AdvancedCalibrationConfig& EightPointAlgorithm::getConfig() const {
    return config_;
}

cv::Mat EightPointAlgorithm::estimateFundamentalMatrix(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    std::vector<bool>& inlier_mask) {

    cv::Mat fundamental_matrix;
    std::vector<uchar> opencv_mask;

    if (config_.use_ransac) {
        fundamental_matrix = cv::findFundamentalMat(
            points1, points2, cv::FM_RANSAC,
            config_.ransac_threshold, config_.ransac_confidence, opencv_mask);
    } else {
        fundamental_matrix = cv::findFundamentalMat(
            points1, points2, cv::FM_8POINT, 3.0, 0.99, opencv_mask);
    }

    // Convert OpenCV mask to bool vector
    inlier_mask.resize(opencv_mask.size());
    std::transform(opencv_mask.begin(), opencv_mask.end(), inlier_mask.begin(),
                  [](uchar x) { return x != 0; });

    return fundamental_matrix;
}

cv::Mat EightPointAlgorithm::fundamentalToEssential(
    const cv::Mat& fundamental_matrix,
    const cv::Mat& camera_matrix1,
    const cv::Mat& camera_matrix2) {

    // E = K2^T * F * K1
    cv::Mat essential_matrix = camera_matrix2.t() * fundamental_matrix * camera_matrix1;

    // Enforce essential matrix constraints: two equal singular values, one zero
    cv::Mat U, S, Vt;
    cv::SVD::compute(essential_matrix, S, U, Vt);

    // Set singular values to [1, 1, 0]
    S.at<double>(0) = 1.0;
    S.at<double>(1) = 1.0;
    S.at<double>(2) = 0.0;

    essential_matrix = U * cv::Mat::diag(S) * Vt;

    return essential_matrix;
}

std::vector<std::pair<cv::Mat, cv::Vec3d>> EightPointAlgorithm::decomposeEssentialMatrix(
    const cv::Mat& essential_matrix) {

    cv::Mat U, S, Vt;
    cv::SVD::compute(essential_matrix, S, U, Vt);

    // Ensure proper rotation matrices
    if (cv::determinant(U) < 0) U *= -1;
    if (cv::determinant(Vt) < 0) Vt *= -1;

    // W matrix for rotation decomposition
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);

    // Four possible solutions
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;
    cv::Vec3d t1(U.at<double>(0, 2), U.at<double>(1, 2), U.at<double>(2, 2));
    cv::Vec3d t2 = -t1;

    std::vector<std::pair<cv::Mat, cv::Vec3d>> poses;
    poses.emplace_back(R1, t1);
    poses.emplace_back(R1, t2);
    poses.emplace_back(R2, t1);
    poses.emplace_back(R2, t2);

    return poses;
}

std::pair<cv::Mat, cv::Vec3d> EightPointAlgorithm::selectCorrectPose(
    const std::vector<std::pair<cv::Mat, cv::Vec3d>>& poses,
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cv::Mat& camera_matrix1,
    const cv::Mat& camera_matrix2,
    std::vector<cv::Point3f>& triangulated_points) {

    int best_pose_index = -1;
    int max_points_in_front = 0;
    std::vector<cv::Point3f> best_triangulated_points;

    for (size_t i = 0; i < poses.size(); ++i) {
        const auto& [R, t] = poses[i];

        // Create projection matrices
        cv::Mat P1 = camera_matrix1 * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat rt = (cv::Mat_<double>(3, 4) <<
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t[0],
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t[1],
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t[2]);
        cv::Mat P2 = camera_matrix2 * rt;

        // Triangulate points
        cv::Mat points_4d;
        cv::triangulatePoints(P1, P2, points1, points2, points_4d);

        // Count points with positive depth in both cameras
        int points_in_front = 0;
        std::vector<cv::Point3f> current_triangulated_points;

        for (int j = 0; j < points_4d.cols; ++j) {
            cv::Vec4f homogeneous_point = points_4d.col(j);
            if (std::abs(homogeneous_point[3]) > 1e-6) {
                cv::Point3f point_3d(
                    homogeneous_point[0] / homogeneous_point[3],
                    homogeneous_point[1] / homogeneous_point[3],
                    homogeneous_point[2] / homogeneous_point[3]
                );

                // Check if point is in front of both cameras
                if (point_3d.z > 0) {
                    // Transform to second camera coordinate system
                    cv::Mat point_cam2 = R * cv::Mat(cv::Vec3f(point_3d.x, point_3d.y, point_3d.z)) +
                                         cv::Mat(cv::Vec3d(t[0], t[1], t[2]));
                    if (point_cam2.at<double>(2) > 0) {
                        points_in_front++;
                        current_triangulated_points.push_back(point_3d);
                    }
                }
            }
        }

        if (points_in_front > max_points_in_front) {
            max_points_in_front = points_in_front;
            best_pose_index = i;
            best_triangulated_points = current_triangulated_points;
        }
    }

    triangulated_points = best_triangulated_points;
    return poses[best_pose_index];
}

double EightPointAlgorithm::calculateReprojectionError(
    const std::vector<cv::Point3f>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    const cv::Mat& camera_matrix,
    const cv::Mat& rotation_matrix,
    const cv::Vec3d& translation_vector) {

    if (points_3d.empty()) return std::numeric_limits<double>::max();

    // Project 3D points to 2D
    std::vector<cv::Point2f> projected_points;
    cv::Vec3d rvec;
    cv::Rodrigues(rotation_matrix, rvec);
    cv::projectPoints(points_3d, rvec, translation_vector, camera_matrix, cv::Mat(), projected_points);

    // Calculate RMS error
    double total_error = 0.0;
    int valid_points = 0;

    for (size_t i = 0; i < std::min(projected_points.size(), points_2d.size()); ++i) {
        double dx = projected_points[i].x - points_2d[i].x;
        double dy = projected_points[i].y - points_2d[i].y;
        total_error += dx * dx + dy * dy;
        valid_points++;
    }

    return valid_points > 0 ? std::sqrt(total_error / valid_points) : std::numeric_limits<double>::max();
}

} // namespace calibration
} // namespace stereo_vision
