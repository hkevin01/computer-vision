#include "camera_calibration.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>

namespace stereo_vision {

CameraCalibration::CameraCalibration() : square_size_(0.0f) {}

CameraCalibration::~CameraCalibration() {}

CameraCalibration::CameraParameters CameraCalibration::calibrateSingleCamera(
    const std::vector<cv::Mat>& images,
    const cv::Size& board_size,
    float square_size) {
    
    board_size_ = board_size;
    square_size_ = square_size;
    
    std::vector<std::vector<cv::Point2f>> image_points;
    generateObjectPoints(board_size, square_size);
    
    // Detect corners in all images
    for (const auto& image : images) {
        std::vector<cv::Point2f> corners;
        if (detectCorners(image, board_size, corners)) {
            image_points.push_back(corners);
        }
    }
    
    if (image_points.size() < 10) {
        throw std::runtime_error("Not enough valid calibration images. Need at least 10.");
    }
    
    CameraParameters params;
    params.image_size = images[0].size();
    
    // Calibrate camera
    std::vector<cv::Mat> rvecs, tvecs;
    params.reprojection_error = cv::calibrateCamera(
        object_points_,
        image_points,
        params.image_size,
        params.camera_matrix,
        params.distortion_coeffs,
        rvecs,
        tvecs
    );
    
    std::cout << "Single camera calibration completed. Reprojection error: " 
              << params.reprojection_error << std::endl;
    
    return params;
}

CameraCalibration::StereoParameters CameraCalibration::calibrateStereoCamera(
    const std::vector<cv::Mat>& left_images,
    const std::vector<cv::Mat>& right_images,
    const cv::Size& board_size,
    float square_size) {
    
    if (left_images.size() != right_images.size()) {
        throw std::runtime_error("Number of left and right images must be equal");
    }
    
    board_size_ = board_size;
    square_size_ = square_size;
    
    std::vector<std::vector<cv::Point2f>> left_image_points, right_image_points;
    generateObjectPoints(board_size, square_size);
    
    // Detect corners in all image pairs
    for (size_t i = 0; i < left_images.size(); ++i) {
        std::vector<cv::Point2f> left_corners, right_corners;
        
        if (detectCorners(left_images[i], board_size, left_corners) &&
            detectCorners(right_images[i], board_size, right_corners)) {
            left_image_points.push_back(left_corners);
            right_image_points.push_back(right_corners);
        }
    }
    
    if (left_image_points.size() < 10) {
        throw std::runtime_error("Not enough valid stereo calibration images. Need at least 10.");
    }
    
    StereoParameters stereo_params;
    cv::Size image_size = left_images[0].size();
    
    // Individual camera calibration
    std::vector<cv::Mat> rvecs1, tvecs1, rvecs2, tvecs2;
    
    stereo_params.left_camera.reprojection_error = cv::calibrateCamera(
        object_points_, left_image_points, image_size,
        stereo_params.left_camera.camera_matrix,
        stereo_params.left_camera.distortion_coeffs,
        rvecs1, tvecs1
    );
    
    stereo_params.right_camera.reprojection_error = cv::calibrateCamera(
        object_points_, right_image_points, image_size,
        stereo_params.right_camera.camera_matrix,
        stereo_params.right_camera.distortion_coeffs,
        rvecs2, tvecs2
    );
    
    stereo_params.left_camera.image_size = image_size;
    stereo_params.right_camera.image_size = image_size;
    
    // Stereo calibration
    stereo_params.reprojection_error = cv::stereoCalibrate(
        object_points_,
        left_image_points,
        right_image_points,
        stereo_params.left_camera.camera_matrix,
        stereo_params.left_camera.distortion_coeffs,
        stereo_params.right_camera.camera_matrix,
        stereo_params.right_camera.distortion_coeffs,
        image_size,
        stereo_params.R,
        stereo_params.T,
        stereo_params.E,
        stereo_params.F,
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5)
    );
    
    // Stereo rectification
    cv::stereoRectify(
        stereo_params.left_camera.camera_matrix,
        stereo_params.left_camera.distortion_coeffs,
        stereo_params.right_camera.camera_matrix,
        stereo_params.right_camera.distortion_coeffs,
        image_size,
        stereo_params.R,
        stereo_params.T,
        stereo_params.R1,
        stereo_params.R2,
        stereo_params.P1,
        stereo_params.P2,
        stereo_params.Q,
        cv::CALIB_ZERO_DISPARITY,
        1,
        image_size,
        &stereo_params.left_roi,
        &stereo_params.right_roi
    );
    
    std::cout << "Stereo calibration completed. Reprojection error: " 
              << stereo_params.reprojection_error << std::endl;
    
    return stereo_params;
}

bool CameraCalibration::detectCorners(const cv::Mat& image, const cv::Size& board_size,
                                     std::vector<cv::Point2f>& corners) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    bool found = cv::findChessboardCorners(gray, board_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    
    if (found) {
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    
    return found;
}

void CameraCalibration::generateObjectPoints(const cv::Size& board_size, float square_size) {
    std::vector<cv::Point3f> corners;
    
    for (int i = 0; i < board_size.height; ++i) {
        for (int j = 0; j < board_size.width; ++j) {
            corners.push_back(cv::Point3f(j * square_size, i * square_size, 0.0f));
        }
    }
    
    object_points_.clear();
    object_points_.resize(1, corners);
}

bool CameraCalibration::saveCalibration(const std::string& filename, 
                                       const StereoParameters& params) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        return false;
    }
    
    fs << "left_camera_matrix" << params.left_camera.camera_matrix;
    fs << "left_distortion" << params.left_camera.distortion_coeffs;
    fs << "right_camera_matrix" << params.right_camera.camera_matrix;
    fs << "right_distortion" << params.right_camera.distortion_coeffs;
    fs << "R" << params.R;
    fs << "T" << params.T;
    fs << "E" << params.E;
    fs << "F" << params.F;
    fs << "R1" << params.R1;
    fs << "R2" << params.R2;
    fs << "P1" << params.P1;
    fs << "P2" << params.P2;
    fs << "Q" << params.Q;
    fs << "reprojection_error" << params.reprojection_error;
    
    return true;
}

bool CameraCalibration::loadCalibration(const std::string& filename, 
                                       StereoParameters& params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    
    fs["left_camera_matrix"] >> params.left_camera.camera_matrix;
    fs["left_distortion"] >> params.left_camera.distortion_coeffs;
    fs["right_camera_matrix"] >> params.right_camera.camera_matrix;
    fs["right_distortion"] >> params.right_camera.distortion_coeffs;
    fs["R"] >> params.R;
    fs["T"] >> params.T;
    fs["E"] >> params.E;
    fs["F"] >> params.F;
    fs["R1"] >> params.R1;
    fs["R2"] >> params.R2;
    fs["P1"] >> params.P1;
    fs["P2"] >> params.P2;
    fs["Q"] >> params.Q;
    fs["reprojection_error"] >> params.reprojection_error;
    
    return true;
}

} // namespace stereo_vision
