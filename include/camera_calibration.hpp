#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace stereo_vision {

/**
 * @brief Camera calibration class for stereo vision
 */
class CameraCalibration {
public:
    struct CameraParameters {
        cv::Mat camera_matrix;
        cv::Mat distortion_coeffs;
        cv::Size image_size;
        double reprojection_error;
    };

    struct StereoParameters {
        CameraParameters left_camera;
        CameraParameters right_camera;
        cv::Mat R, T, E, F;  // Rotation, Translation, Essential, Fundamental matrices
        cv::Mat R1, R2, P1, P2, Q;  // Rectification matrices
        cv::Rect left_roi, right_roi;
        double reprojection_error;
    };

public:
    CameraCalibration();
    ~CameraCalibration();

    /**
     * @brief Calibrate a single camera using checkerboard images
     * @param images Vector of calibration images
     * @param board_size Size of the checkerboard (inner corners)
     * @param square_size Physical size of checkerboard squares in mm
     * @return Camera parameters
     */
    CameraParameters calibrateSingleCamera(
        const std::vector<cv::Mat>& images,
        const cv::Size& board_size,
        float square_size
    );

    /**
     * @brief Calibrate stereo camera pair
     * @param left_images Left camera calibration images
     * @param right_images Right camera calibration images
     * @param board_size Size of the checkerboard (inner corners)
     * @param square_size Physical size of checkerboard squares in mm
     * @return Stereo parameters
     */
    StereoParameters calibrateStereoCamera(
        const std::vector<cv::Mat>& left_images,
        const std::vector<cv::Mat>& right_images,
        const cv::Size& board_size,
        float square_size
    );

    /**
     * @brief Save calibration parameters to file
     */
    bool saveCalibration(const std::string& filename, const StereoParameters& params);

    /**
     * @brief Load calibration parameters from file
     */
    bool loadCalibration(const std::string& filename, StereoParameters& params);

    /**
     * @brief Detect checkerboard corners in image
     */
    bool detectCorners(const cv::Mat& image, const cv::Size& board_size, 
                      std::vector<cv::Point2f>& corners);

private:
    cv::Size board_size_;
    float square_size_;
    std::vector<std::vector<cv::Point3f>> object_points_;
    
    void generateObjectPoints(const cv::Size& board_size, float square_size);
};

} // namespace stereo_vision
