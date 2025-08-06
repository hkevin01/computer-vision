#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <map>

namespace stereo_vision {
namespace calibration {

/**
 * @brief Advanced camera calibration algorithms
 *
 * This module implements sophisticated calibration methods including:
 * - Longuet-Higgins' eight-point algorithm for fundamental matrix estimation
 * - Tsai's two-stage calibration with radial alignment constraint (RAC)
 * - Integration with existing OpenCV-based calibration workflows
 */

/**
 * @brief Result structure for eight-point algorithm calibration
 */
struct EightPointResult {
    cv::Mat fundamental_matrix;      ///< 3x3 fundamental matrix F
    cv::Mat essential_matrix;        ///< 3x3 essential matrix E
    cv::Mat rotation_matrix;         ///< 3x3 rotation matrix R
    cv::Vec3d translation_vector;    ///< Translation vector t
    std::vector<cv::Point3f> triangulated_points; ///< 3D points from triangulation
    double reprojection_error;      ///< RMS reprojection error
    double estimation_error;        ///< Estimation error for algorithm comparison
    int inlier_count;               ///< Number of inlier correspondences
    std::vector<bool> inlier_mask;  ///< Mask indicating inlier correspondences
    double computation_time;        ///< Computation time in milliseconds
    bool success;                   ///< Whether calibration succeeded
    std::string error_message;      ///< Error description if failed
};

/**
 * @brief Result structure for Tsai calibration
 */
struct TsaiCalibrationResult {
    cv::Mat camera_matrix;          ///< 3x3 intrinsic camera matrix K
    cv::Mat distortion_coeffs;      ///< Distortion coefficients [k1, k2, p1, p2, k3]
    cv::Mat rotation_matrix;        ///< 3x3 rotation matrix R
    cv::Vec3d translation_vector;   ///< Translation vector t
    double focal_length_x;          ///< Focal length in x direction (pixels)
    double focal_length_y;          ///< Focal length in y direction (pixels)
    cv::Point2d principal_point;    ///< Principal point (cx, cy)
    double radial_distortion_k1;    ///< First radial distortion coefficient
    double radial_distortion_k2;    ///< Second radial distortion coefficient
    double tangential_p1;           ///< First tangential distortion coefficient
    double tangential_p2;           ///< Second tangential distortion coefficient
    double reprojection_error;      ///< RMS reprojection error
    double computation_time;        ///< Computation time in milliseconds
    bool success;                   ///< Whether calibration succeeded
    std::string error_message;      ///< Error description if failed
};

/**
 * @brief Configuration for advanced calibration algorithms
 */
struct AdvancedCalibrationConfig {
    // Eight-point algorithm settings
    bool use_ransac = true;                    ///< Use RANSAC for robust estimation
    double ransac_threshold = 1.0;            ///< RANSAC threshold in pixels
    double ransac_confidence = 0.99;          ///< RANSAC confidence level
    int max_iterations = 1000;                ///< Maximum RANSAC iterations

    // Tsai calibration settings
    bool estimate_radial_distortion = true;   ///< Estimate radial distortion
    bool estimate_tangential_distortion = false; ///< Estimate tangential distortion
    int optimization_iterations = 100;        ///< Nonlinear optimization iterations
    double optimization_tolerance = 1e-6;     ///< Optimization convergence tolerance

    // General settings
    bool verbose = false;                      ///< Enable verbose output
    double pixel_error_threshold = 2.0;       ///< Maximum allowed pixel error
    int min_points_required = 8;              ///< Minimum points for calibration
};

/**
 * @brief Progress callback function type
 */
using ProgressCallback = std::function<void(const std::string& message, double progress)>;

/**
 * @brief Eight-Point Algorithm Implementation
 *
 * Implements Longuet-Higgins' eight-point algorithm for fundamental matrix estimation
 * and camera pose recovery from point correspondences.
 */
class EightPointAlgorithm {
public:
    explicit EightPointAlgorithm(const AdvancedCalibrationConfig& config = AdvancedCalibrationConfig());

    /**
     * @brief Estimate fundamental matrix from point correspondences
     * @param points1 Points in first image (left camera)
     * @param points2 Corresponding points in second image (right camera)
     * @param image_size Size of the images
     * @param progress_callback Optional callback for progress updates
     * @return Eight-point calibration result
     */
    EightPointResult calibrate(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2,
        const cv::Size& image_size,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Estimate pose from point correspondences (alias for calibrate)
     * @param points1 Points in first image
     * @param points2 Corresponding points in second image
     * @param progress_callback Optional callback for progress updates
     * @return Eight-point calibration result
     */
    EightPointResult estimatePose(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Estimate fundamental matrix with known camera matrices
     * @param points1 Points in first image
     * @param points2 Corresponding points in second image
     * @param camera_matrix1 Intrinsic matrix of first camera
     * @param camera_matrix2 Intrinsic matrix of second camera
     * @param progress_callback Optional callback for progress updates
     * @return Eight-point calibration result with essential matrix
     */
    EightPointResult calibrateWithIntrinsics(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2,
        const cv::Mat& camera_matrix1,
        const cv::Mat& camera_matrix2,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Set configuration parameters
     */
    void setConfig(const AdvancedCalibrationConfig& config);

    /**
     * @brief Get current configuration
     */
    const AdvancedCalibrationConfig& getConfig() const;

private:
    AdvancedCalibrationConfig config_;

    // Core algorithm methods
    cv::Mat estimateFundamentalMatrix(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2,
        std::vector<bool>& inlier_mask
    );

    cv::Mat fundamentalToEssential(
        const cv::Mat& fundamental_matrix,
        const cv::Mat& camera_matrix1,
        const cv::Mat& camera_matrix2
    );

    std::vector<std::pair<cv::Mat, cv::Vec3d>> decomposeEssentialMatrix(
        const cv::Mat& essential_matrix
    );

    std::pair<cv::Mat, cv::Vec3d> selectCorrectPose(
        const std::vector<std::pair<cv::Mat, cv::Vec3d>>& poses,
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2,
        const cv::Mat& camera_matrix1,
        const cv::Mat& camera_matrix2,
        std::vector<cv::Point3f>& triangulated_points
    );

    double calculateReprojectionError(
        const std::vector<cv::Point3f>& points_3d,
        const std::vector<cv::Point2f>& points_2d,
        const cv::Mat& camera_matrix,
        const cv::Mat& rotation_matrix,
        const cv::Vec3d& translation_vector
    );
};

/**
 * @brief Tsai's Two-Stage Calibration Implementation
 *
 * Implements Tsai's two-stage calibration algorithm based on the
 * Radial Alignment Constraint (RAC) for handling radial distortion.
 */
class TsaiCalibration {
public:
    explicit TsaiCalibration(const AdvancedCalibrationConfig& config = AdvancedCalibrationConfig());

    /**
     * @brief Perform Tsai's two-stage calibration
     * @param object_points 3D object points in world coordinates
     * @param image_points Corresponding 2D image points
     * @param image_size Size of the calibration images
     * @param progress_callback Optional callback for progress updates
     * @return Tsai calibration result
     */
    TsaiCalibrationResult calibrate(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const cv::Size& image_size,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Single view calibration with known world points
     * @param object_points 3D object points for single view
     * @param image_points Corresponding 2D image points
     * @param image_size Size of the calibration image
     * @param progress_callback Optional callback for progress updates
     * @return Tsai calibration result
     */
    TsaiCalibrationResult calibrateSingleView(
        const std::vector<cv::Point3f>& object_points,
        const std::vector<cv::Point2f>& image_points,
        const cv::Size& image_size,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Set configuration parameters
     */
    void setConfig(const AdvancedCalibrationConfig& config);

    /**
     * @brief Get current configuration
     */
    const AdvancedCalibrationConfig& getConfig() const;

private:
    AdvancedCalibrationConfig config_;

    // Stage 1: Extrinsic parameter estimation using RAC
    std::pair<cv::Mat, cv::Vec3d> stageOneExtrinsicEstimation(
        const std::vector<cv::Point3f>& object_points,
        const std::vector<cv::Point2f>& image_points,
        const cv::Size& image_size
    );

    // Stage 2: Nonlinear optimization of all parameters
    TsaiCalibrationResult stageTwoNonlinearOptimization(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const cv::Size& image_size,
        const cv::Mat& initial_rotation,
        const cv::Vec3d& initial_translation
    );

    // Radial alignment constraint implementation
    double radialAlignmentError(
        const cv::Point3f& object_point,
        const cv::Point2f& image_point,
        const cv::Mat& rotation_matrix,
        const cv::Vec3d& translation_vector,
        const cv::Mat& camera_matrix,
        const cv::Mat& distortion_coeffs
    );

    // Nonlinear optimization objective function
    void optimizationObjective(
        const std::vector<double>& parameters,
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        std::vector<double>& residuals
    ) const;

    // Parameter vector management
    std::vector<double> parametersToVector(
        const cv::Mat& camera_matrix,
        const cv::Mat& distortion_coeffs,
        const std::vector<cv::Mat>& rotation_matrices,
        const std::vector<cv::Vec3d>& translation_vectors
    ) const;

    void vectorToParameters(
        const std::vector<double>& parameters,
        cv::Mat& camera_matrix,
        cv::Mat& distortion_coeffs,
        std::vector<cv::Mat>& rotation_matrices,
        std::vector<cv::Vec3d>& translation_vectors
    ) const;
};

/**
 * @brief Analysis results from algorithm comparison
 */
struct CalibrationAnalysis {
    std::string best_algorithm;
    double best_reprojection_error;
    double mean_reprojection_error;
    double std_reprojection_error;
    double mean_computation_time;
    double std_computation_time;
    std::string consistency;
    std::string parameter_consistency;
    std::string summary;
};

/**
 * @brief Comparison results for different calibration algorithms
 */
struct CalibrationComparison {
    TsaiCalibrationResult opencv_result;
    TsaiCalibrationResult tsai_result;
    EightPointResult eight_point_result;
    CalibrationAnalysis analysis;
    std::string recommendation;
    bool success;
    std::string error_message;
};

/**
 * @brief Advanced Calibration Manager
 *
 * High-level interface for managing different calibration algorithms
 * and integrating with existing calibration workflows.
 */
class AdvancedCalibrationManager {
public:
    explicit AdvancedCalibrationManager(const AdvancedCalibrationConfig& config = AdvancedCalibrationConfig());
    ~AdvancedCalibrationManager();

    /**
     * @brief Compare different calibration algorithms
     * @param object_points Object points for calibration
     * @param image_points Image points for calibration
     * @param image_size Size of calibration images
     * @param progress_callback Optional progress callback
     * @return Comparison results for different algorithms
     */
    CalibrationComparison compareAlgorithms(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const cv::Size& image_size,
        ProgressCallback progress_callback = nullptr
    );

    /**
     * @brief Set configuration parameters
     */
    void setConfig(const AdvancedCalibrationConfig& config);

    /**
     * @brief Get current configuration
     */
    const AdvancedCalibrationConfig& getConfig() const;

private:
    AdvancedCalibrationConfig config_;
    std::unique_ptr<EightPointAlgorithm> eight_point_algorithm_;
    std::unique_ptr<TsaiCalibration> tsai_calibration_;

    // Helper methods
    TsaiCalibrationResult runOpenCVCalibration(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const cv::Size& image_size
    );

    CalibrationAnalysis analyzeResults(const CalibrationComparison& comparison);
    std::string generateRecommendation(const CalibrationComparison& comparison);
    void printComparisonSummary(const CalibrationComparison& comparison);
    double calculateStandardDeviation(const std::vector<double>& values, double mean);
};

} // namespace calibration
} // namespace stereo_vision
