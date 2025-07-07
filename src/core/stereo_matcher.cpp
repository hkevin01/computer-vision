#include "stereo_matcher.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace stereo_vision {

StereoMatcher::StereoMatcher() : is_initialized_(false), use_cuda_(false) {}

StereoMatcher::~StereoMatcher() {}

bool StereoMatcher::initialize(const CameraCalibration::StereoParameters& stereo_params) {
    stereo_params_ = stereo_params;
    
    // Initialize rectification maps
    cv::initUndistortRectifyMap(
        stereo_params.left_camera.camera_matrix,
        stereo_params.left_camera.distortion_coeffs,
        stereo_params.R1,
        stereo_params.P1,
        stereo_params.left_camera.image_size,
        CV_16SC2,
        map1_left_,
        map2_left_
    );
    
    cv::initUndistortRectifyMap(
        stereo_params.right_camera.camera_matrix,
        stereo_params.right_camera.distortion_coeffs,
        stereo_params.R2,
        stereo_params.P2,
        stereo_params.right_camera.image_size,
        CV_16SC2,
        map1_right_,
        map2_right_
    );
    
    // Initialize stereo matcher
    if (params_.use_cuda) {
        initializeCudaMatcher();
    } else {
        initializeCpuMatcher();
    }
    
    is_initialized_ = true;
    return true;
}

void StereoMatcher::initializeCpuMatcher() {
    stereo_matcher_ = cv::StereoSGBM::create(
        params_.min_disparity,
        params_.num_disparities,
        params_.block_size,
        params_.P1,
        params_.P2,
        params_.disp12_max_diff,
        params_.pre_filter_cap,
        params_.uniqueness_ratio,
        params_.speckle_window_size,
        params_.speckle_range,
        cv::StereoSGBM::MODE_SGBM
    );
    use_cuda_ = false;
}

void StereoMatcher::initializeCudaMatcher() {
    // Note: CUDA stereo matcher would be implemented here
    // For now, fall back to CPU implementation
    std::cout << "CUDA stereo matching not yet implemented, using CPU" << std::endl;
    initializeCpuMatcher();
}

cv::Mat StereoMatcher::computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image) {
    if (!is_initialized_) {
        throw std::runtime_error("StereoMatcher not initialized. Call initialize() first.");
    }
    
    cv::Mat left_rect, right_rect;
    rectifyImages(left_image, right_image, left_rect, right_rect);
    
    cv::Mat disparity;
    stereo_matcher_->compute(left_rect, right_rect, disparity);
    
    // Convert to proper format and post-process
    cv::Mat disparity_8u;
    disparity.convertTo(disparity_8u, CV_8U, 255.0 / (params_.num_disparities * 16.0));
    
    return postProcessDisparity(disparity);
}

void StereoMatcher::rectifyImages(const cv::Mat& left_raw, const cv::Mat& right_raw,
                                 cv::Mat& left_rect, cv::Mat& right_rect) {
    cv::remap(left_raw, left_rect, map1_left_, map2_left_, cv::INTER_LINEAR);
    cv::remap(right_raw, right_rect, map1_right_, map2_right_, cv::INTER_LINEAR);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr StereoMatcher::generatePointCloud(
    const cv::Mat& disparity, const cv::Mat& left_image) {
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Convert disparity to 3D points
    cv::Mat points_3d;
    cv::reprojectImageTo3D(disparity, points_3d, stereo_params_.Q);
    
    cv::Mat left_rect;
    cv::remap(left_image, left_rect, map1_left_, map2_left_, cv::INTER_LINEAR);
    
    // Convert to point cloud
    for (int y = 0; y < points_3d.rows; ++y) {
        for (int x = 0; x < points_3d.cols; ++x) {
            cv::Vec3f point = points_3d.at<cv::Vec3f>(y, x);
            
            // Filter out invalid points
            if (point[2] > 0 && point[2] < 10000) {  // Reasonable depth range
                pcl::PointXYZRGB pcl_point;
                pcl_point.x = point[0] / 1000.0f;  // Convert mm to meters
                pcl_point.y = point[1] / 1000.0f;
                pcl_point.z = point[2] / 1000.0f;
                
                // Add color information
                if (left_rect.channels() == 3) {
                    cv::Vec3b color = left_rect.at<cv::Vec3b>(y, x);
                    pcl_point.r = color[2];
                    pcl_point.g = color[1];
                    pcl_point.b = color[0];
                } else {
                    uint8_t intensity = left_rect.at<uint8_t>(y, x);
                    pcl_point.r = pcl_point.g = pcl_point.b = intensity;
                }
                
                cloud->points.push_back(pcl_point);
            }
        }
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    
    return cloud;
}

cv::Mat StereoMatcher::postProcessDisparity(const cv::Mat& disparity) {
    cv::Mat filtered;
    
    // Apply median filter to reduce noise
    cv::medianBlur(disparity, filtered, 5);
    
    // Fill holes (simple approach)
    cv::Mat mask = (filtered == 0);
    cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    filtered.setTo(cv::Scalar(0), mask);
    
    return filtered;
}

void StereoMatcher::setParameters(const MatchingParameters& params) {
    params_ = params;
    if (is_initialized_) {
        if (params_.use_cuda) {
            initializeCudaMatcher();
        } else {
            initializeCpuMatcher();
        }
    }
}

StereoMatcher::MatchingParameters StereoMatcher::getParameters() const {
    return params_;
}

} // namespace stereo_vision
