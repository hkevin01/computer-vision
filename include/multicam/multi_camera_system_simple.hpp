#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <opencv2/opencv.hpp>

namespace stereovision {
namespace multicam {

enum class SynchronizationMode {
    HARDWARE_SYNC = 0,
    SOFTWARE_SYNC = 1,
    TIMESTAMP_SYNC = 2
};

enum class ProcessingMode {
    DEPTH_ESTIMATION = 0,
    POINT_CLOUD = 1,
    FULL_3D = 2
};

struct CameraConfig {
    int camera_id;
    cv::Size resolution;
    double fps;
    bool enable_auto_exposure;
    double exposure_time;

    CameraConfig() : camera_id(-1), resolution(640, 480), fps(30.0),
                    enable_auto_exposure(true), exposure_time(0.033) {}
};

struct CameraIntrinsics {
    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
    cv::Size image_size;

    CameraIntrinsics() : image_size(640, 480) {
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        distortion_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    }
};

struct StereoExtrinsics {
    cv::Mat rotation;
    cv::Mat translation;
    cv::Mat essential_matrix;
    cv::Mat fundamental_matrix;

    StereoExtrinsics() {
        rotation = cv::Mat::eye(3, 3, CV_64F);
        translation = cv::Mat::zeros(3, 1, CV_64F);
        essential_matrix = cv::Mat::zeros(3, 3, CV_64F);
        fundamental_matrix = cv::Mat::zeros(3, 3, CV_64F);
    }
};

class MultiCameraSystem {
public:
    MultiCameraSystem();
    ~MultiCameraSystem();

    // Camera management
    bool addCamera(int camera_id, const CameraConfig& config);
    bool removeCamera(int camera_id);
    bool isConnected(int camera_id) const;
    std::vector<int> getConnectedCameras() const;

    // Synchronization
    // (See extended declaration later for captureSynchronizedFrames with stats update)
    bool setSynchronizationMode(SynchronizationMode mode);
    SynchronizationMode getSynchronizationMode() const { return sync_mode_; }

    // Configuration
    bool setCameraConfig(int camera_id, const CameraConfig& config);
    CameraConfig getCameraConfig(int camera_id) const;

    // Statistics
    double getAverageFrameRate() const;
    std::map<int, double> getCameraFrameRates() const;

    struct SyncStats {
        std::atomic<uint64_t> frame_pairs{0};
        std::atomic<double> avg_delta_ms{0.0};
        std::atomic<double> max_delta_ms{0.0};
        std::atomic<double> jitter_ms{0.0};
        std::atomic<double> last_delta_ms{0.0};
        std::atomic<uint64_t> disconnect_events{0};
        std::atomic<uint64_t> dropped_frames{0};
        std::atomic<uint64_t> consecutive_failures{0};
    };

    enum class SyncQuality { EXCELLENT, GOOD, POOR, UNKNOWN };

    const SyncStats& getSyncStats() const { return sync_stats_; }
    SyncQuality classifySyncQuality() const;
    void resetSyncStats();
    bool hadRecentDisconnect() const { return recent_disconnect_flag_.load(); }
    void clearRecentDisconnectFlag() { recent_disconnect_flag_.store(false); }

    // Extended capture with internal stats update
    bool captureSynchronizedFrames(std::map<int, cv::Mat>& frames,
                                   std::map<int, std::chrono::high_resolution_clock::time_point>& timestamps);

private:
    std::map<int, cv::VideoCapture> cameras_;
    std::map<int, CameraConfig> camera_configs_;
    mutable std::mutex cameras_mutex_;
    SynchronizationMode sync_mode_;
    std::atomic<bool> is_capturing_;
    SyncStats sync_stats_;
    std::atomic<bool> recent_disconnect_flag_{false};
    std::chrono::high_resolution_clock::time_point last_sync_reset_ {std::chrono::high_resolution_clock::now()};

    bool initializeCamera(int camera_id, const CameraConfig& config);
    void applyCameraSettings(int camera_id, const CameraConfig& config);
    void updateSyncStats(const std::map<int, std::chrono::high_resolution_clock::time_point>& timestamps);
    void recordDisconnect();
};

class MultiCameraCalibrator {
public:
    MultiCameraCalibrator();

    // Calibration data collection
    bool addCalibrationFrame(const std::map<int, cv::Mat>& frames);
    bool setChessboardPattern(cv::Size pattern_size, float square_size);

    // Calibration execution
    bool calibrateCameras();
    bool calibrateStereoSystem(int camera1_id, int camera2_id);

    // Results
    CameraIntrinsics getCameraIntrinsics(int camera_id) const;
    StereoExtrinsics getStereoExtrinsics(int camera1_id, int camera2_id) const;

    // Validation
    double getCalibrationError(int camera_id) const;
    double getStereoCalibrationError(int camera1_id, int camera2_id) const;

    // Save/Load calibration
    bool saveCalibration(const std::string& filename) const;
    bool loadCalibration(const std::string& filename);

private:
    std::map<int, std::vector<std::vector<cv::Point2f>>> image_points_;
    std::vector<std::vector<cv::Point3f>> object_points_;
    std::map<int, CameraIntrinsics> camera_intrinsics_;
    std::map<std::pair<int, int>, StereoExtrinsics> stereo_extrinsics_;

    cv::Size chessboard_size_;
    float square_size_;
    bool pattern_set_;

    std::vector<cv::Point3f> generateChessboardPoints();
    bool detectChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners);
};

class RealtimeMultiCameraProcessor {
public:
    RealtimeMultiCameraProcessor(std::shared_ptr<MultiCameraSystem> camera_system);
    ~RealtimeMultiCameraProcessor();

    // Processing control
    bool startProcessing(ProcessingMode mode = ProcessingMode::DEPTH_ESTIMATION);
    void stopProcessing();
    bool isProcessing() const { return is_processing_; }

    // Configuration
    void setTargetFPS(double fps) { target_fps_ = fps; }
    void setProcessingMode(ProcessingMode mode) { processing_mode_ = mode; }

    // Results access
    bool getLatestResults(std::map<std::pair<int, int>, cv::Mat>& depth_maps,
                         std::map<std::pair<int, int>, cv::Mat>& point_clouds);

    // Performance monitoring
    double getCurrentFPS() const;
    size_t getProcessingQueueSize() const;

private:
    std::shared_ptr<MultiCameraSystem> camera_system_;
    std::atomic<bool> is_processing_;
    std::thread processing_thread_;
    ProcessingMode processing_mode_;
    double target_fps_;

    // Results storage
    std::map<std::pair<int, int>, cv::Mat> latest_depth_maps_;
    std::map<std::pair<int, int>, cv::Mat> latest_point_clouds_;
    mutable std::mutex results_mutex_;

    // Processing methods
    void processingLoop();
    void processFrames(const std::map<int, cv::Mat>& frames);
    cv::Mat computeDepthMap(const cv::Mat& left_image, const cv::Mat& right_image,
                           int camera1_id, int camera2_id);
    cv::Mat computePointCloud(const cv::Mat& left_image, const cv::Mat& right_image,
                             int camera1_id, int camera2_id);
};

// Utility functions
class MultiCameraUtils {
public:
    // Auto-detection
    static std::vector<int> detectAvailableCameras();
    static bool testCameraConnection(int camera_id);

    // Synchronization testing
    static bool testSynchronization(const std::vector<int>& camera_ids, int num_frames = 10);
    static double measureSynchronizationError(const std::map<int, std::chrono::high_resolution_clock::time_point>& timestamps);

    // Quality assessment
    static double assessImageQuality(const cv::Mat& image);
    static bool validateStereoConfiguration(int camera1_id, int camera2_id, const std::map<int, cv::Mat>& frames);
};

} // namespace multicam
} // namespace stereovision
