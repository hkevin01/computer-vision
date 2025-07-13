#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>

namespace stereovision {
namespace multicam {

/**
 * @brief Multi-camera stereo vision system supporting 2+ synchronized cameras
 * 
 * This class manages multiple cameras for advanced stereo vision applications,
 * including:
 * - Synchronized capture from multiple cameras
 * - Multi-baseline stereo for improved accuracy
 * - Camera array calibration
 * - Distributed processing across camera pairs
 */
class MultiCameraSystem {
public:
    enum class CameraRole {
        LEFT_PRIMARY,       ///< Primary left camera
        RIGHT_PRIMARY,      ///< Primary right camera
        LEFT_SECONDARY,     ///< Secondary left camera for multi-baseline
        RIGHT_SECONDARY,    ///< Secondary right camera for multi-baseline
        AUXILIARY,          ///< Additional camera for enhanced coverage
        REFERENCE           ///< Reference camera for synchronization
    };

    enum class SyncMode {
        HARDWARE_SYNC,      ///< Hardware-based synchronization (requires special cameras)
        SOFTWARE_SYNC,      ///< Software-based synchronization
        TRIGGER_SYNC,       ///< External trigger synchronization
        BEST_EFFORT         ///< Best effort synchronization
    };

    struct CameraConfig {
        int camera_id = 0;
        CameraRole role = CameraRole::AUXILIARY;
        cv::Size resolution = cv::Size(1920, 1080);
        double fps = 30.0;
        std::string device_path;
        cv::Mat camera_matrix;
        cv::Mat distortion_coeffs;
        cv::Mat rotation_matrix;      ///< Rotation relative to reference camera
        cv::Mat translation_vector;   ///< Translation relative to reference camera
        bool enabled = true;
        
        // Advanced settings
        double exposure_time = -1;    ///< Auto exposure if -1
        double gain = -1;             ///< Auto gain if -1
        bool auto_white_balance = true;
    };

    struct SynchronizedFrames {
        std::vector<cv::Mat> frames;
        std::vector<std::chrono::high_resolution_clock::time_point> timestamps;
        std::vector<int> camera_ids;
        double max_time_diff_ms = 0.0;  ///< Maximum time difference between frames
        bool is_synchronized = false;
        uint64_t frame_number = 0;
    };

    struct MultiCameraCalibration {
        std::vector<cv::Mat> camera_matrices;
        std::vector<cv::Mat> distortion_coeffs;
        std::vector<cv::Mat> rotation_matrices;     ///< Relative to reference camera
        std::vector<cv::Mat> translation_vectors;   ///< Relative to reference camera
        std::map<std::pair<int, int>, cv::Mat> essential_matrices;
        std::map<std::pair<int, int>, cv::Mat> fundamental_matrices;
        double reprojection_error = 0.0;
        bool is_valid = false;
    };

    struct ProcessingStats {
        double capture_fps = 0.0;
        double processing_fps = 0.0;
        double sync_accuracy_ms = 0.0;
        size_t dropped_frames = 0;
        size_t total_frames = 0;
        std::vector<double> per_camera_fps;
        double memory_usage_mb = 0.0;
    };

public:
    explicit MultiCameraSystem(SyncMode sync_mode = SyncMode::SOFTWARE_SYNC);
    ~MultiCameraSystem();

    // Non-copyable but movable
    MultiCameraSystem(const MultiCameraSystem&) = delete;
    MultiCameraSystem& operator=(const MultiCameraSystem&) = delete;
    MultiCameraSystem(MultiCameraSystem&&) = default;
    MultiCameraSystem& operator=(MultiCameraSystem&&) = default;

    /**
     * @brief Add a camera to the system
     * @param config Camera configuration
     * @return Camera index in the system
     */
    int addCamera(const CameraConfig& config);

    /**
     * @brief Remove a camera from the system
     * @param camera_index Index of camera to remove
     */
    bool removeCamera(int camera_index);

    /**
     * @brief Initialize all cameras and start capture threads
     */
    bool initialize();

    /**
     * @brief Start synchronized capture
     */
    bool startCapture();

    /**
     * @brief Stop synchronized capture
     */
    void stopCapture();

    /**
     * @brief Capture synchronized frames from all cameras
     * @param timeout_ms Timeout in milliseconds
     * @return Synchronized frame set
     */
    SynchronizedFrames captureSynchronized(int timeout_ms = 1000);

    /**
     * @brief Get the latest synchronized frames (non-blocking)
     */
    SynchronizedFrames getLatestFrames();

    /**
     * @brief Calibrate the multi-camera system
     * @param calibration_images Vector of image sets (one per camera)
     * @param pattern_size Calibration pattern size
     * @param square_size Physical size of calibration squares
     * @return Calibration results
     */
    MultiCameraCalibration calibrateSystem(
        const std::vector<std::vector<cv::Mat>>& calibration_images,
        cv::Size pattern_size,
        float square_size);

    /**
     * @brief Load calibration from file
     */
    bool loadCalibration(const std::string& calibration_file);

    /**
     * @brief Save calibration to file
     */
    bool saveCalibration(const std::string& calibration_file) const;

    /**
     * @brief Generate multi-baseline stereo disparity maps
     * @param frames Synchronized frame set
     * @return Vector of disparity maps from different camera pairs
     */
    std::vector<cv::Mat> computeMultiBaselineDisparity(const SynchronizedFrames& frames);

    /**
     * @brief Fuse multiple disparity maps for improved accuracy
     * @param disparity_maps Vector of disparity maps
     * @param confidence_maps Corresponding confidence maps
     * @return Fused disparity map
     */
    cv::Mat fuseDisparityMaps(const std::vector<cv::Mat>& disparity_maps,
                             const std::vector<cv::Mat>& confidence_maps);

    /**
     * @brief Generate 3D point cloud from multi-camera system
     * @param frames Synchronized frame set
     * @return Enhanced 3D point cloud
     */
    // Generate point cloud from synchronized frames
    cv::Mat generatePointCloud(const SynchronizedFrames& frames);

    /**
     * @brief Get current processing statistics
     */
    ProcessingStats getStats() const;

    /**
     * @brief Get camera configuration
     */
    const std::vector<CameraConfig>& getCameraConfigs() const { return camera_configs_; }

    /**
     * @brief Get current calibration
     */
    const MultiCameraCalibration& getCalibration() const { return calibration_; }

    /**
     * @brief Check if system is capturing
     */
    bool isCapturing() const { return capturing_; }

    /**
     * @brief Get number of cameras
     */
    size_t getCameraCount() const { return camera_configs_.size(); }

    /**
     * @brief Set synchronization tolerance
     * @param tolerance_ms Maximum allowed time difference between frames
     */
    void setSyncTolerance(double tolerance_ms) { sync_tolerance_ms_ = tolerance_ms; }

    /**
     * @brief Enable/disable specific camera
     */
    void enableCamera(int camera_index, bool enabled);

    /**
     * @brief Get camera roles for stereo pair selection
     */
    std::vector<std::pair<int, int>> getStereoPairs() const;

private:
    // Camera management
    std::vector<CameraConfig> camera_configs_;
    std::vector<std::unique_ptr<cv::VideoCapture>> cameras_;
    MultiCameraCalibration calibration_;

    // Synchronization
    SyncMode sync_mode_;
    double sync_tolerance_ms_ = 50.0;  // 50ms tolerance by default
    
    // Capture threading
    std::vector<std::thread> capture_threads_;
    std::atomic<bool> capturing_{false};
    std::atomic<bool> should_stop_{false};
    
    // Frame buffer and synchronization
    struct FrameBuffer {
        std::vector<cv::Mat> frames;
        std::vector<std::chrono::high_resolution_clock::time_point> timestamps;
        std::mutex mutex;
        std::condition_variable condition;
        size_t max_buffer_size = 10;
        uint64_t frame_counter = 0;
    };
    
    std::vector<FrameBuffer> frame_buffers_;
    mutable std::mutex stats_mutex_;
    ProcessingStats stats_;

    // Private methods
    void captureThread(int camera_index);
    SynchronizedFrames findSynchronizedFrames();
    void updateStats();
    bool validateCameraConfig(const CameraConfig& config) const;
    std::vector<cv::Point3f> generateCalibrationPoints(cv::Size pattern_size, float square_size) const;
    cv::Mat computeDisparityForPair(const cv::Mat& left, const cv::Mat& right, 
                                   const cv::Mat& Q_matrix);
    void cleanupOldFrames();
};

/**
 * @brief Utility class for multi-camera calibration
 */
class MultiCameraCalibrator {
public:
    struct CalibrationPattern {
        enum Type { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES };
        Type type = CHESSBOARD;
        cv::Size size = cv::Size(9, 6);
        float square_size = 25.0f; // mm
    };

    /**
     * @brief Detect calibration pattern in multiple cameras simultaneously
     */
    static bool detectPatternMultiCamera(
        const std::vector<cv::Mat>& images,
        const CalibrationPattern& pattern,
        std::vector<std::vector<cv::Point2f>>& corners);

    /**
     * @brief Perform bundle adjustment for multi-camera system
     */
    static MultiCameraSystem::MultiCameraCalibration bundleAdjustment(
        const std::vector<std::vector<std::vector<cv::Point2f>>>& image_points,
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<cv::Size>& image_sizes);

    /**
     * @brief Validate calibration quality
     */
    static double validateCalibration(
        const MultiCameraSystem::MultiCameraCalibration& calibration,
        const std::vector<std::vector<std::vector<cv::Point2f>>>& image_points,
        const std::vector<std::vector<cv::Point3f>>& object_points);
};

/**
 * @brief Real-time multi-camera processor
 */
class RealtimeMultiCameraProcessor {
public:
    struct ProcessingConfig {
        bool enable_disparity;
        bool enable_point_cloud;
        bool enable_object_detection;
        int target_fps;
        cv::Size processing_resolution;
        int max_disparity;
        bool use_gpu;
        
        ProcessingConfig() 
            : enable_disparity(true)
            , enable_point_cloud(true)
            , enable_object_detection(false)
            , target_fps(30)
            , processing_resolution(640, 480)
            , max_disparity(128)
            , use_gpu(true) {}
    };

    explicit RealtimeMultiCameraProcessor(const ProcessingConfig& config = ProcessingConfig());

    /**
     * @brief Process synchronized frames in real-time
     */
    struct ProcessingResults {
        std::vector<cv::Mat> disparity_maps;
        cv::Mat point_cloud;
        std::vector<cv::Rect> detected_objects;
        double processing_time_ms;
        bool success;
    };

    ProcessingResults process(const MultiCameraSystem::SynchronizedFrames& frames,
                            const MultiCameraSystem::MultiCameraCalibration& calibration);

    /**
     * @brief Get processing statistics
     */
    struct ProcessingStats {
        double avg_processing_time_ms;
        double current_fps;
        size_t frames_processed;
        size_t frames_dropped;
        double gpu_utilization;
    };

    ProcessingStats getProcessingStats() const;

private:
    ProcessingConfig config_;
    ProcessingStats stats_;
    
    // GPU processing components (when available)
    std::unique_ptr<class GPUStereoProcessor> gpu_processor_;
    std::unique_ptr<class GPUPointCloudGenerator> gpu_point_cloud_generator_;
};

} // namespace stereo_vision::multicam
