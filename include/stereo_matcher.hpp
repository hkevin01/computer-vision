#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "camera_calibration.hpp"

namespace stereo_vision {

/**
 * @brief Stereo matching and depth estimation class
 */
class StereoMatcher {
public:
    struct MatchingParameters {
        int min_disparity = 0;
        int num_disparities = 64;
        int block_size = 11;
        int P1 = 8;
        int P2 = 32;
        int disp12_max_diff = 1;
        int pre_filter_cap = 63;
        int uniqueness_ratio = 10;
        int speckle_window_size = 100;
        int speckle_range = 32;
        bool use_cuda = true;
    };

public:
    StereoMatcher();
    ~StereoMatcher();

    /**
     * @brief Initialize stereo matcher with calibration parameters
     */
    bool initialize(const CameraCalibration::StereoParameters& stereo_params);

    /**
     * @brief Compute disparity map from stereo image pair
     */
    cv::Mat computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image);

    /**
     * @brief Generate 3D point cloud from disparity map
     */
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr generatePointCloud(
        const cv::Mat& disparity,
        const cv::Mat& left_image
    );

    /**
     * @brief Set matching parameters
     */
    void setParameters(const MatchingParameters& params);

    /**
     * @brief Get current matching parameters
     */
    MatchingParameters getParameters() const;

    /**
     * @brief Rectify stereo image pair
     */
    void rectifyImages(const cv::Mat& left_raw, const cv::Mat& right_raw,
                      cv::Mat& left_rect, cv::Mat& right_rect);

private:
    MatchingParameters params_;
    CameraCalibration::StereoParameters stereo_params_;
    
    cv::Ptr<cv::StereoSGBM> stereo_matcher_;
    cv::Mat map1_left_, map2_left_, map1_right_, map2_right_;
    
    bool is_initialized_;
    bool use_cuda_;

    void initializeCudaMatcher();
    void initializeCpuMatcher();
    cv::Mat postProcessDisparity(const cv::Mat& disparity);
};

} // namespace stereo_vision
