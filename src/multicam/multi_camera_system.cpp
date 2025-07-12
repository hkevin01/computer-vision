#include "multicam/multi_camera_system.hpp"
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>

namespace stereovision {
namespace multicam {

// ============================================================================
// MultiCameraSystem Implementation
// ============================================================================

MultiCameraSystem::MultiCameraSystem() 
    : sync_mode_(SyncMode::SOFTWARE)
    , frame_timeout_ms_(1000)
    , max_sync_offset_ms_(50)
    , is_capturing_(false) {}

MultiCameraSystem::~MultiCameraSystem() {
    stopCapture();
}

bool MultiCameraSystem::addCamera(int camera_id, const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    
    if (cameras_.find(camera_id) != cameras_.end()) {
        return false; // Camera already exists
    }
    
    auto camera = std::make_unique<cv::VideoCapture>(camera_id);
    if (!camera->isOpened()) {
        return false;
    }
    
    // Apply configuration
    if (config.width > 0 && config.height > 0) {
        camera->set(cv::CAP_PROP_FRAME_WIDTH, config.width);
        camera->set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
    }
    
    if (config.fps > 0) {
        camera->set(cv::CAP_PROP_FPS, config.fps);
    }
    
    cameras_[camera_id] = std::move(camera);
    camera_configs_[camera_id] = config;
    
    return true;
}

bool MultiCameraSystem::removeCamera(int camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    
    auto it = cameras_.find(camera_id);
    if (it == cameras_.end()) {
        return false;
    }
    
    cameras_.erase(it);
    camera_configs_.erase(camera_id);
    
    return true;
}

std::vector<int> MultiCameraSystem::getCameraIds() const {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    std::vector<int> ids;
    for (const auto& pair : cameras_) {
        ids.push_back(pair.first);
    }
    return ids;
}

bool MultiCameraSystem::startCapture() {
    if (is_capturing_) {
        return false;
    }
    
    if (cameras_.empty()) {
        return false;
    }
    
    is_capturing_ = true;
    
    // Start capture thread for each camera
    for (const auto& pair : cameras_) {
        int camera_id = pair.first;
        capture_threads_[camera_id] = std::thread(
            &MultiCameraSystem::captureLoop, this, camera_id);
    }
    
    return true;
}

void MultiCameraSystem::stopCapture() {
    is_capturing_ = false;
    
    // Wait for all threads to finish
    for (auto& pair : capture_threads_) {
        if (pair.second.joinable()) {
            pair.second.join();
        }
    }
    capture_threads_.clear();
}

bool MultiCameraSystem::captureSynchronizedFrames(
    std::map<int, cv::Mat>& frames, 
    std::map<int, std::chrono::steady_clock::time_point>& timestamps) {
    
    if (!is_capturing_ || cameras_.empty()) {
        return false;
    }
    
    frames.clear();
    timestamps.clear();
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Collect frames from all cameras
    std::map<int, FrameData> frame_data;
    {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        frame_condition_.wait_for(lock, std::chrono::milliseconds(frame_timeout_ms_),
            [this]() { return latest_frames_.size() == cameras_.size(); });
        
        for (const auto& pair : latest_frames_) {
            frame_data[pair.first] = pair.second;
        }
    }
    
    if (frame_data.size() != cameras_.size()) {
        return false; // Not all cameras provided frames
    }
    
    // Check synchronization
    if (sync_mode_ == SyncMode::SOFTWARE) {
        auto reference_time = frame_data.begin()->second.timestamp;
        
        for (const auto& pair : frame_data) {
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::abs((pair.second.timestamp - reference_time).count()));
            
            if (time_diff.count() > max_sync_offset_ms_) {
                return false; // Frames not synchronized
            }
        }
    }
    
    // Extract frames and timestamps
    for (const auto& pair : frame_data) {
        frames[pair.first] = pair.second.frame.clone();
        timestamps[pair.first] = pair.second.timestamp;
    }
    
    return true;
}

void MultiCameraSystem::setSyncMode(SyncMode mode) {
    sync_mode_ = mode;
}

void MultiCameraSystem::setFrameTimeout(int timeout_ms) {
    frame_timeout_ms_ = timeout_ms;
}

void MultiCameraSystem::setMaxSyncOffset(int offset_ms) {
    max_sync_offset_ms_ = offset_ms;
}

void MultiCameraSystem::captureLoop(int camera_id) {
    auto& camera = cameras_[camera_id];
    cv::Mat frame;
    
    while (is_capturing_) {
        if (camera->read(frame)) {
            auto timestamp = std::chrono::steady_clock::now();
            
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                latest_frames_[camera_id] = {frame.clone(), timestamp};
            }
            frame_condition_.notify_all();
        }
        
        // Small delay to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// ============================================================================
// MultiCameraCalibrator Implementation
// ============================================================================

MultiCameraCalibrator::MultiCameraCalibrator(
    cv::Size board_size, float square_size)
    : board_size_(board_size)
    , square_size_(square_size)
    , calibration_flags_(cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE) {}

bool MultiCameraCalibrator::addCalibrationData(
    const std::map<int, cv::Mat>& images) {
    
    if (images.size() < 2) {
        return false; // Need at least 2 cameras
    }
    
    std::map<int, std::vector<cv::Point2f>> camera_corners;
    
    // Find chessboard corners in all images
    for (const auto& pair : images) {
        int camera_id = pair.first;
        const cv::Mat& image = pair.second;
        
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            image, board_size_, corners, calibration_flags_);
        
        if (!found) {
            return false; // All cameras must see the pattern
        }
        
        // Refine corner positions
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        
        camera_corners[camera_id] = corners;
        
        // Store image size for camera matrix initialization
        if (image_sizes_.find(camera_id) == image_sizes_.end()) {
            image_sizes_[camera_id] = image.size();
        }
    }
    
    // Store calibration data
    calibration_data_.push_back(camera_corners);
    
    return true;
}

bool MultiCameraCalibrator::calibrate() {
    if (calibration_data_.size() < 10) {
        return false; // Need sufficient calibration data
    }
    
    // Generate 3D object points
    std::vector<cv::Point3f> object_points;
    for (int i = 0; i < board_size_.height; ++i) {
        for (int j = 0; j < board_size_.width; ++j) {
            object_points.push_back(cv::Point3f(
                j * square_size_, i * square_size_, 0));
        }
    }
    
    // Calibrate each camera individually
    for (const auto& size_pair : image_sizes_) {
        int camera_id = size_pair.first;
        cv::Size image_size = size_pair.second;
        
        // Collect image points for this camera
        std::vector<std::vector<cv::Point2f>> image_points;
        std::vector<std::vector<cv::Point3f>> object_points_vec;
        
        for (const auto& data : calibration_data_) {
            auto it = data.find(camera_id);
            if (it != data.end()) {
                image_points.push_back(it->second);
                object_points_vec.push_back(object_points);
            }
        }
        
        if (image_points.size() < 5) {
            continue; // Need minimum data for calibration
        }
        
        cv::Mat camera_matrix, dist_coeffs;
        std::vector<cv::Mat> rvecs, tvecs;
        
        double rms = cv::calibrateCamera(
            object_points_vec, image_points, image_size,
            camera_matrix, dist_coeffs, rvecs, tvecs);
        
        CalibrationResult result;
        result.camera_matrix = camera_matrix;
        result.dist_coeffs = dist_coeffs;
        result.rms_error = rms;
        
        calibration_results_[camera_id] = result;
    }
    
    // Perform stereo calibration for camera pairs
    calibrateStereoRigs();
    
    return !calibration_results_.empty();
}

CalibrationResult MultiCameraCalibrator::getCalibrationResult(int camera_id) const {
    auto it = calibration_results_.find(camera_id);
    if (it != calibration_results_.end()) {
        return it->second;
    }
    return CalibrationResult();
}

std::map<std::pair<int, int>, StereoCalibrationResult> 
MultiCameraCalibrator::getStereoResults() const {
    return stereo_results_;
}

void MultiCameraCalibrator::calibrateStereoRigs() {
    std::vector<int> camera_ids;
    for (const auto& pair : calibration_results_) {
        camera_ids.push_back(pair.first);
    }
    
    // Generate 3D object points
    std::vector<cv::Point3f> object_points;
    for (int i = 0; i < board_size_.height; ++i) {
        for (int j = 0; j < board_size_.width; ++j) {
            object_points.push_back(cv::Point3f(
                j * square_size_, i * square_size_, 0));
        }
    }
    
    // Calibrate each camera pair
    for (size_t i = 0; i < camera_ids.size(); ++i) {
        for (size_t j = i + 1; j < camera_ids.size(); ++j) {
            int camera1_id = camera_ids[i];
            int camera2_id = camera_ids[j];
            
            // Collect synchronized image points
            std::vector<std::vector<cv::Point2f>> image_points1, image_points2;
            std::vector<std::vector<cv::Point3f>> object_points_vec;
            
            for (const auto& data : calibration_data_) {
                auto it1 = data.find(camera1_id);
                auto it2 = data.find(camera2_id);
                
                if (it1 != data.end() && it2 != data.end()) {
                    image_points1.push_back(it1->second);
                    image_points2.push_back(it2->second);
                    object_points_vec.push_back(object_points);
                }
            }
            
            if (object_points_vec.size() < 5) {
                continue; // Need minimum data
            }
            
            cv::Mat R, T, E, F;
            cv::Size image_size = image_sizes_.at(camera1_id);
            
            double rms = cv::stereoCalibrate(
                object_points_vec, image_points1, image_points2,
                calibration_results_[camera1_id].camera_matrix,
                calibration_results_[camera1_id].dist_coeffs,
                calibration_results_[camera2_id].camera_matrix,
                calibration_results_[camera2_id].dist_coeffs,
                image_size, R, T, E, F);
            
            StereoCalibrationResult stereo_result;
            stereo_result.rotation = R;
            stereo_result.translation = T;
            stereo_result.essential_matrix = E;
            stereo_result.fundamental_matrix = F;
            stereo_result.rms_error = rms;
            
            stereo_results_[std::make_pair(camera1_id, camera2_id)] = stereo_result;
        }
    }
}

// ============================================================================
// RealtimeMultiCameraProcessor Implementation
// ============================================================================

RealtimeMultiCameraProcessor::RealtimeMultiCameraProcessor()
    : is_processing_(false)
    , target_fps_(30)
    , processing_mode_(ProcessingMode::DEPTH_ESTIMATION) {}

RealtimeMultiCameraProcessor::~RealtimeMultiCameraProcessor() {
    stopProcessing();
}

bool RealtimeMultiCameraProcessor::initialize(
    std::shared_ptr<MultiCameraSystem> camera_system) {
    
    camera_system_ = camera_system;
    return camera_system_ != nullptr;
}

void RealtimeMultiCameraProcessor::setProcessingMode(ProcessingMode mode) {
    processing_mode_ = mode;
}

void RealtimeMultiCameraProcessor::setTargetFPS(double fps) {
    target_fps_ = fps;
}

void RealtimeMultiCameraProcessor::startProcessing() {
    if (is_processing_ || !camera_system_) {
        return;
    }
    
    is_processing_ = true;
    processing_thread_ = std::thread(&RealtimeMultiCameraProcessor::processingLoop, this);
}

void RealtimeMultiCameraProcessor::stopProcessing() {
    is_processing_ = false;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

bool RealtimeMultiCameraProcessor::getLatestResults(
    std::map<std::pair<int, int>, cv::Mat>& depth_maps,
    std::map<std::pair<int, int>, cv::Mat>& point_clouds) {
    
    std::lock_guard<std::mutex> lock(results_mutex_);
    
    depth_maps = latest_depth_maps_;
    point_clouds = latest_point_clouds_;
    
    return !depth_maps.empty() || !point_clouds.empty();
}

void RealtimeMultiCameraProcessor::processingLoop() {
    auto target_frame_time = std::chrono::microseconds(
        static_cast<long long>(1000000.0 / target_fps_));
    
    while (is_processing_) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Capture synchronized frames
        std::map<int, cv::Mat> frames;
        std::map<int, std::chrono::steady_clock::time_point> timestamps;
        
        if (camera_system_->captureSynchronizedFrames(frames, timestamps)) {
            processFrames(frames);
        }
        
        // Maintain target frame rate
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed < target_frame_time) {
            std::this_thread::sleep_for(target_frame_time - elapsed);
        }
    }
}

void RealtimeMultiCameraProcessor::processFrames(
    const std::map<int, cv::Mat>& frames) {
    
    std::vector<int> camera_ids;
    for (const auto& pair : frames) {
        camera_ids.push_back(pair.first);
    }
    
    std::map<std::pair<int, int>, cv::Mat> depth_maps;
    std::map<std::pair<int, int>, cv::Mat> point_clouds;
    
    // Process each camera pair
    for (size_t i = 0; i < camera_ids.size(); ++i) {
        for (size_t j = i + 1; j < camera_ids.size(); ++j) {
            int camera1_id = camera_ids[i];
            int camera2_id = camera_ids[j];
            
            auto key = std::make_pair(camera1_id, camera2_id);
            
            const cv::Mat& img1 = frames.at(camera1_id);
            const cv::Mat& img2 = frames.at(camera2_id);
            
            if (processing_mode_ == ProcessingMode::DEPTH_ESTIMATION ||
                processing_mode_ == ProcessingMode::FULL_3D) {
                
                cv::Mat depth_map = computeDepthMap(img1, img2);
                if (!depth_map.empty()) {
                    depth_maps[key] = depth_map;
                }
            }
            
            if (processing_mode_ == ProcessingMode::POINT_CLOUD ||
                processing_mode_ == ProcessingMode::FULL_3D) {
                
                cv::Mat point_cloud = computePointCloud(img1, img2);
                if (!point_cloud.empty()) {
                    point_clouds[key] = point_cloud;
                }
            }
        }
    }
    
    // Update results
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        latest_depth_maps_ = depth_maps;
        latest_point_clouds_ = point_clouds;
    }
}

cv::Mat RealtimeMultiCameraProcessor::computeDepthMap(
    const cv::Mat& img1, const cv::Mat& img2) {
    
    // Simple stereo matching using OpenCV's built-in algorithm
    cv::Mat gray1, gray2;
    
    if (img1.channels() == 3) {
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = img1;
        gray2 = img2;
    }
    
    auto stereo = cv::StereoBM::create(16, 21);
    cv::Mat disparity;
    stereo->compute(gray1, gray2, disparity);
    
    // Convert disparity to depth (simplified)
    cv::Mat depth;
    disparity.convertTo(depth, CV_32F, 1.0/16.0);
    
    return depth;
}

cv::Mat RealtimeMultiCameraProcessor::computePointCloud(
    const cv::Mat& img1, const cv::Mat& img2) {
    
    cv::Mat depth_map = computeDepthMap(img1, img2);
    if (depth_map.empty()) {
        return cv::Mat();
    }
    
    // Generate point cloud from depth map (simplified)
    std::vector<cv::Point3f> points;
    
    for (int y = 0; y < depth_map.rows; ++y) {
        for (int x = 0; x < depth_map.cols; ++x) {
            float depth = depth_map.at<float>(y, x);
            if (depth > 0) {
                // Simple pinhole camera model (would use actual calibration in practice)
                float fx = 525.0f, fy = 525.0f;
                float cx = depth_map.cols / 2.0f, cy = depth_map.rows / 2.0f;
                
                cv::Point3f point;
                point.z = depth;
                point.x = (x - cx) * depth / fx;
                point.y = (y - cy) * depth / fy;
                
                points.push_back(point);
            }
        }
    }
    
    // Convert to OpenCV Mat format
    cv::Mat point_cloud(points.size(), 1, CV_32FC3);
    std::memcpy(point_cloud.data, points.data(), points.size() * sizeof(cv::Point3f));
    
    return point_cloud;
}

} // namespace multicam
} // namespace stereovision
