#include "include/multicam/multi_camera_system_simple.hpp"
#include <iostream>
#include <algorithm>
#include <thread>

namespace stereovision {
namespace multicam {

MultiCameraSystem::MultiCameraSystem() 
    : sync_mode_(SynchronizationMode::SOFTWARE_SYNC), is_capturing_(false) {
}

MultiCameraSystem::~MultiCameraSystem() {
    is_capturing_ = false;
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    cameras_.clear();
}

bool MultiCameraSystem::addCamera(int camera_id, const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    
    if (cameras_.find(camera_id) != cameras_.end()) {
        std::cout << "Camera " << camera_id << " already exists\n";
        return false;
    }
    
    if (initializeCamera(camera_id, config)) {
        camera_configs_[camera_id] = config;
        std::cout << "Successfully added camera " << camera_id << std::endl;
        return true;
    }
    
    return false;
}

bool MultiCameraSystem::removeCamera(int camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    
    auto it = cameras_.find(camera_id);
    if (it != cameras_.end()) {
        it->second.release();
        cameras_.erase(it);
        camera_configs_.erase(camera_id);
        return true;
    }
    
    return false;
}

bool MultiCameraSystem::isConnected(int camera_id) const {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    auto it = cameras_.find(camera_id);
    return it != cameras_.end() && it->second.isOpened();
}

std::vector<int> MultiCameraSystem::getConnectedCameras() const {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    std::vector<int> connected;
    
    for (const auto& pair : cameras_) {
        if (pair.second.isOpened()) {
            connected.push_back(pair.first);
        }
    }
    
    return connected;
}

bool MultiCameraSystem::captureSynchronizedFrames(
    std::map<int, cv::Mat>& frames, 
    std::map<int, std::chrono::high_resolution_clock::time_point>& timestamps) {
    
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    frames.clear();
    timestamps.clear();
    
    if (cameras_.empty()) {
        return false;
    }
    
    // For software synchronization, capture as quickly as possible
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (auto& pair : cameras_) {
        cv::Mat frame;
        if (pair.second.read(frame)) {
            frames[pair.first] = frame.clone();
            timestamps[pair.first] = std::chrono::high_resolution_clock::now();
        }
    }
    
    return !frames.empty();
}

bool MultiCameraSystem::setSynchronizationMode(SynchronizationMode mode) {
    sync_mode_ = mode;
    std::cout << "Set synchronization mode to " << static_cast<int>(mode) << std::endl;
    return true;
}

bool MultiCameraSystem::setCameraConfig(int camera_id, const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    
    auto it = cameras_.find(camera_id);
    if (it != cameras_.end()) {
        applyCameraSettings(camera_id, config);
        camera_configs_[camera_id] = config;
        return true;
    }
    
    return false;
}

CameraConfig MultiCameraSystem::getCameraConfig(int camera_id) const {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    
    auto it = camera_configs_.find(camera_id);
    if (it != camera_configs_.end()) {
        return it->second;
    }
    
    return CameraConfig{};
}

double MultiCameraSystem::getAverageFrameRate() const {
    auto rates = getCameraFrameRates();
    if (rates.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& pair : rates) {
        sum += pair.second;
    }
    
    return sum / rates.size();
}

std::map<int, double> MultiCameraSystem::getCameraFrameRates() const {
    std::lock_guard<std::mutex> lock(cameras_mutex_);
    std::map<int, double> rates;
    
    for (const auto& pair : camera_configs_) {
        rates[pair.first] = pair.second.fps;
    }
    
    return rates;
}

bool MultiCameraSystem::initializeCamera(int camera_id, const CameraConfig& config) {
    cv::VideoCapture cap(camera_id);
    
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << camera_id << std::endl;
        return false;
    }
    
    // Set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, config.resolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.resolution.height);
    cap.set(cv::CAP_PROP_FPS, config.fps);
    
    if (!config.enable_auto_exposure) {
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // Manual exposure mode
        cap.set(cv::CAP_PROP_EXPOSURE, config.exposure_time);
    }
    
    cameras_[camera_id] = std::move(cap);
    return true;
}

void MultiCameraSystem::applyCameraSettings(int camera_id, const CameraConfig& config) {
    auto it = cameras_.find(camera_id);
    if (it != cameras_.end()) {
        it->second.set(cv::CAP_PROP_FRAME_WIDTH, config.resolution.width);
        it->second.set(cv::CAP_PROP_FRAME_HEIGHT, config.resolution.height);
        it->second.set(cv::CAP_PROP_FPS, config.fps);
        
        if (!config.enable_auto_exposure) {
            it->second.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
            it->second.set(cv::CAP_PROP_EXPOSURE, config.exposure_time);
        }
    }
}

// MultiCameraCalibrator implementation
MultiCameraCalibrator::MultiCameraCalibrator() 
    : chessboard_size_(9, 6), square_size_(25.0f), pattern_set_(false) {
}

bool MultiCameraCalibrator::addCalibrationFrame(const std::map<int, cv::Mat>& frames) {
    if (!pattern_set_) {
        std::cerr << "Chessboard pattern not set\n";
        return false;
    }
    
    std::map<int, std::vector<cv::Point2f>> frame_corners;
    bool all_detected = true;
    
    for (const auto& pair : frames) {
        std::vector<cv::Point2f> corners;
        if (detectChessboardCorners(pair.second, corners)) {
            frame_corners[pair.first] = corners;
        } else {
            all_detected = false;
            std::cout << "Failed to detect corners in camera " << pair.first << std::endl;
        }
    }
    
    if (all_detected && !frame_corners.empty()) {
        // Add to calibration data
        for (const auto& pair : frame_corners) {
            image_points_[pair.first].push_back(pair.second);
        }
        
        // Add object points (same for all cameras)
        if (object_points_.empty() || object_points_.size() < image_points_.begin()->second.size()) {
            object_points_.push_back(generateChessboardPoints());
        }
        
        std::cout << "Added calibration frame. Total frames: " << object_points_.size() << std::endl;
        return true;
    }
    
    return false;
}

bool MultiCameraCalibrator::setChessboardPattern(cv::Size pattern_size, float square_size) {
    chessboard_size_ = pattern_size;
    square_size_ = square_size;
    pattern_set_ = true;
    
    std::cout << "Set chessboard pattern: " << pattern_size.width << "x" << pattern_size.height 
              << ", square size: " << square_size << "mm" << std::endl;
    return true;
}

bool MultiCameraCalibrator::calibrateCameras() {
    if (object_points_.empty() || image_points_.empty()) {
        std::cerr << "No calibration data available\n";
        return false;
    }
    
    for (const auto& pair : image_points_) {
        int camera_id = pair.first;
        const auto& points = pair.second;
        
        if (points.size() < 10) {
            std::cerr << "Need at least 10 calibration frames for camera " << camera_id << std::endl;
            continue;
        }
        
        CameraIntrinsics intrinsics;
        std::vector<cv::Mat> rvecs, tvecs;
        
        // Get image size from first image points
        intrinsics.image_size = cv::Size(640, 480); // Default, should be from actual images
        
        double rms = cv::calibrateCamera(
            object_points_, points, intrinsics.image_size,
            intrinsics.camera_matrix, intrinsics.distortion_coeffs,
            rvecs, tvecs
        );
        
        camera_intrinsics_[camera_id] = intrinsics;
        
        std::cout << "Camera " << camera_id << " calibrated with RMS error: " << rms << std::endl;
    }
    
    return !camera_intrinsics_.empty();
}

bool MultiCameraCalibrator::calibrateStereoSystem(int camera1_id, int camera2_id) {
    if (camera_intrinsics_.find(camera1_id) == camera_intrinsics_.end() ||
        camera_intrinsics_.find(camera2_id) == camera_intrinsics_.end()) {
        std::cerr << "Both cameras must be calibrated first\n";
        return false;
    }
    
    if (image_points_.find(camera1_id) == image_points_.end() ||
        image_points_.find(camera2_id) == image_points_.end()) {
        std::cerr << "No image points for stereo calibration\n";
        return false;
    }
    
    const auto& points1 = image_points_[camera1_id];
    const auto& points2 = image_points_[camera2_id];
    
    if (points1.size() != points2.size()) {
        std::cerr << "Mismatched number of calibration frames\n";
        return false;
    }
    
    const auto& intrinsics1 = camera_intrinsics_[camera1_id];
    const auto& intrinsics2 = camera_intrinsics_[camera2_id];
    
    StereoExtrinsics extrinsics;
    
    double rms = cv::stereoCalibrate(
        object_points_, points1, points2,
        intrinsics1.camera_matrix, intrinsics1.distortion_coeffs,
        intrinsics2.camera_matrix, intrinsics2.distortion_coeffs,
        intrinsics1.image_size,
        extrinsics.rotation, extrinsics.translation,
        extrinsics.essential_matrix, extrinsics.fundamental_matrix
    );
    
    stereo_extrinsics_[{camera1_id, camera2_id}] = extrinsics;
    
    std::cout << "Stereo system (" << camera1_id << ", " << camera2_id 
              << ") calibrated with RMS error: " << rms << std::endl;
    
    return true;
}

CameraIntrinsics MultiCameraCalibrator::getCameraIntrinsics(int camera_id) const {
    auto it = camera_intrinsics_.find(camera_id);
    if (it != camera_intrinsics_.end()) {
        return it->second;
    }
    return CameraIntrinsics{};
}

StereoExtrinsics MultiCameraCalibrator::getStereoExtrinsics(int camera1_id, int camera2_id) const {
    auto it = stereo_extrinsics_.find({camera1_id, camera2_id});
    if (it != stereo_extrinsics_.end()) {
        return it->second;
    }
    return StereoExtrinsics{};
}

double MultiCameraCalibrator::getCalibrationError(int camera_id) const {
    // Simplified - in real implementation would recalculate projection error
    return 0.5; // Simulated error
}

double MultiCameraCalibrator::getStereoCalibrationError(int camera1_id, int camera2_id) const {
    // Simplified - in real implementation would recalculate stereo projection error
    return 0.3; // Simulated error
}

bool MultiCameraCalibrator::saveCalibration(const std::string& filename) const {
    std::cout << "Saving calibration to " << filename << " (simulation)" << std::endl;
    return true;
}

bool MultiCameraCalibrator::loadCalibration(const std::string& filename) {
    std::cout << "Loading calibration from " << filename << " (simulation)" << std::endl;
    return true;
}

std::vector<cv::Point3f> MultiCameraCalibrator::generateChessboardPoints() {
    std::vector<cv::Point3f> points;
    
    for (int i = 0; i < chessboard_size_.height; ++i) {
        for (int j = 0; j < chessboard_size_.width; ++j) {
            points.push_back(cv::Point3f(j * square_size_, i * square_size_, 0.0f));
        }
    }
    
    return points;
}

bool MultiCameraCalibrator::detectChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    bool found = cv::findChessboardCorners(gray, chessboard_size_, corners);
    
    if (found) {
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    
    return found;
}

// RealtimeMultiCameraProcessor implementation
RealtimeMultiCameraProcessor::RealtimeMultiCameraProcessor(std::shared_ptr<MultiCameraSystem> camera_system)
    : camera_system_(camera_system), is_processing_(false), 
      processing_mode_(ProcessingMode::DEPTH_ESTIMATION), target_fps_(30.0) {
}

RealtimeMultiCameraProcessor::~RealtimeMultiCameraProcessor() {
    stopProcessing();
}

bool RealtimeMultiCameraProcessor::startProcessing(ProcessingMode mode) {
    if (is_processing_) {
        std::cout << "Processing already running\n";
        return false;
    }
    
    processing_mode_ = mode;
    is_processing_ = true;
    
    processing_thread_ = std::thread(&RealtimeMultiCameraProcessor::processingLoop, this);
    
    std::cout << "Started multi-camera processing in mode " << static_cast<int>(mode) << std::endl;
    return true;
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
    auto frame_duration = std::chrono::microseconds(
        static_cast<long long>(1000000.0 / target_fps_));
    
    while (is_processing_) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::map<int, cv::Mat> frames;
        std::map<int, std::chrono::high_resolution_clock::time_point> timestamps;
        
        if (camera_system_->captureSynchronizedFrames(frames, timestamps)) {
            processFrames(frames);
        }
        
        // Maintain target FPS
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = end_time - start_time;
        
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }
    }
}

void RealtimeMultiCameraProcessor::processFrames(const std::map<int, cv::Mat>& frames) {
    std::map<std::pair<int, int>, cv::Mat> depth_maps;
    std::map<std::pair<int, int>, cv::Mat> point_clouds;
    
    // Process all camera pairs
    auto camera_ids = camera_system_->getConnectedCameras();
    
    for (size_t i = 0; i < camera_ids.size(); ++i) {
        for (size_t j = i + 1; j < camera_ids.size(); ++j) {
            int cam1 = camera_ids[i];
            int cam2 = camera_ids[j];
            
            auto it1 = frames.find(cam1);
            auto it2 = frames.find(cam2);
            
            if (it1 != frames.end() && it2 != frames.end()) {
                if (processing_mode_ == ProcessingMode::DEPTH_ESTIMATION ||
                    processing_mode_ == ProcessingMode::FULL_3D) {
                    cv::Mat depth_map = computeDepthMap(it1->second, it2->second, cam1, cam2);
                    depth_maps[{cam1, cam2}] = depth_map;
                }
                
                if (processing_mode_ == ProcessingMode::POINT_CLOUD ||
                    processing_mode_ == ProcessingMode::FULL_3D) {
                    cv::Mat point_cloud = computePointCloud(it1->second, it2->second, cam1, cam2);
                    point_clouds[{cam1, cam2}] = point_cloud;
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
    const cv::Mat& left_image, const cv::Mat& right_image, 
    int camera1_id, int camera2_id) {
    
    // Convert to grayscale if needed
    cv::Mat left_gray, right_gray;
    if (left_image.channels() == 3) {
        cv::cvtColor(left_image, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_image, right_gray, cv::COLOR_BGR2GRAY);
    } else {
        left_gray = left_image;
        right_gray = right_image;
    }
    
    // Use OpenCV's stereo matching for now
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(64, 21);
    
    cv::Mat disparity;
    stereo->compute(left_gray, right_gray, disparity);
    
    // Convert to depth (simplified)
    cv::Mat depth;
    disparity.convertTo(depth, CV_32F, 1.0/16.0);
    
    return depth;
}

cv::Mat RealtimeMultiCameraProcessor::computePointCloud(
    const cv::Mat& left_image, const cv::Mat& right_image,
    int camera1_id, int camera2_id) {
    
    cv::Mat depth = computeDepthMap(left_image, right_image, camera1_id, camera2_id);
    
    // Simple point cloud generation (x, y, z)
    cv::Mat point_cloud(depth.rows, depth.cols, CV_32FC3);
    
    float fx = 500.0f; // Simplified camera parameters
    float fy = 500.0f;
    float cx = depth.cols / 2.0f;
    float cy = depth.rows / 2.0f;
    
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float d = depth.at<float>(y, x);
            if (d > 0) {
                cv::Vec3f& point = point_cloud.at<cv::Vec3f>(y, x);
                point[0] = (x - cx) * d / fx; // X
                point[1] = (y - cy) * d / fy; // Y
                point[2] = d;                 // Z
            } else {
                point_cloud.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
            }
        }
    }
    
    return point_cloud;
}

double RealtimeMultiCameraProcessor::getCurrentFPS() const {
    return target_fps_; // Simplified
}

size_t RealtimeMultiCameraProcessor::getProcessingQueueSize() const {
    return 0; // Simplified
}

// MultiCameraUtils implementation
std::vector<int> MultiCameraUtils::detectAvailableCameras() {
    std::vector<int> available;
    
    for (int i = 0; i < 10; ++i) { // Check first 10 camera indices
        if (testCameraConnection(i)) {
            available.push_back(i);
        }
    }
    
    std::cout << "Found " << available.size() << " available cameras" << std::endl;
    return available;
}

bool MultiCameraUtils::testCameraConnection(int camera_id) {
    cv::VideoCapture cap(camera_id);
    bool connected = cap.isOpened();
    cap.release();
    return connected;
}

bool MultiCameraUtils::testSynchronization(const std::vector<int>& camera_ids, int num_frames) {
    std::cout << "Testing synchronization for " << camera_ids.size() << " cameras" << std::endl;
    
    // Simplified synchronization test
    MultiCameraSystem system;
    
    for (int id : camera_ids) {
        CameraConfig config;
        system.addCamera(id, config);
    }
    
    for (int i = 0; i < num_frames; ++i) {
        std::map<int, cv::Mat> frames;
        std::map<int, std::chrono::high_resolution_clock::time_point> timestamps;
        
        if (system.captureSynchronizedFrames(frames, timestamps)) {
            double error = measureSynchronizationError(timestamps);
            std::cout << "Frame " << i << " sync error: " << error << "ms" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
    
    return true;
}

double MultiCameraUtils::measureSynchronizationError(
    const std::map<int, std::chrono::high_resolution_clock::time_point>& timestamps) {
    
    if (timestamps.size() < 2) return 0.0;
    
    auto first_time = timestamps.begin()->second;
    double max_diff = 0.0;
    
    for (const auto& pair : timestamps) {
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(
            pair.second - first_time).count();
        max_diff = std::max(max_diff, std::abs(diff / 1000.0)); // Convert to milliseconds
    }
    
    return max_diff;
}

double MultiCameraUtils::assessImageQuality(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    return stddev[0] * stddev[0]; // Variance of Laplacian as sharpness measure
}

bool MultiCameraUtils::validateStereoConfiguration(
    int camera1_id, int camera2_id, const std::map<int, cv::Mat>& frames) {
    
    auto it1 = frames.find(camera1_id);
    auto it2 = frames.find(camera2_id);
    
    if (it1 == frames.end() || it2 == frames.end()) {
        return false;
    }
    
    // Simple validation checks
    const cv::Mat& img1 = it1->second;
    const cv::Mat& img2 = it2->second;
    
    // Check if images have same size
    if (img1.size() != img2.size()) {
        std::cerr << "Images have different sizes\n";
        return false;
    }
    
    // Check image quality
    double quality1 = assessImageQuality(img1);
    double quality2 = assessImageQuality(img2);
    
    if (quality1 < 100.0 || quality2 < 100.0) {
        std::cerr << "Low image quality detected\n";
        return false;
    }
    
    std::cout << "Stereo configuration validated successfully\n";
    return true;
}

} // namespace multicam
} // namespace stereovision
