#pragma once

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QDir>
#include <QTimer>
#include <QJsonObject>
#include <QJsonDocument>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <atomic>
#include "stereo_matcher.hpp"
#include "camera_calibration.hpp"
#include "point_cloud_processor.hpp"

namespace stereo_vision::batch {

/**
 * Represents a stereo image pair for batch processing
 */
struct StereoImagePair {
    QString leftImagePath;
    QString rightImagePath;
    QString outputBaseName;
    QDateTime timestamp;
    cv::Size imageSize;
    bool isValid = false;

    // Processing status
    enum Status {
        Pending,
        Processing,
        Completed,
        Failed,
        Skipped
    };
    Status status = Pending;
    QString errorMessage;

    // Processing metrics
    double processingTimeMs = 0.0;
    size_t outputFileSizeBytes = 0;
    int pointCloudPoints = 0;
};

/**
 * Batch processing configuration and settings
 */
struct BatchProcessingConfig {
    // Input/Output settings
    QString inputDirectory;
    QString outputDirectory;
    QString outputFormat = "ply"; // ply, pcd, xyz
    QString namingPattern = "{basename}_{timestamp}"; // Output naming pattern

    // Processing settings
    bool overwriteExisting = false;
    bool resumeInterrupted = true;
    bool validateInputs = true;
    bool generateReport = true;

    // Performance settings
    int maxConcurrentJobs = 4;
    int maxMemoryUsageMB = 4096;
    double timeoutMinutes = 10.0;

    // Quality settings
    bool enableQualityCheck = true;
    double minPointCloudDensity = 0.1; // Minimum points per pixel

    // Stereo processing parameters
    stereo_vision::StereoMatcher::MatchingParameters stereoParams;
    stereo_vision::CameraCalibration::StereoParameters calibrationParams;

    // Progress tracking
    int saveProgressEveryN = 10; // Save progress after every N processed images
};

/**
 * Progress tracking and statistics
 */
struct BatchProcessingProgress {
    // Counters
    std::atomic<int> totalPairs{0};
    std::atomic<int> processedPairs{0};
    std::atomic<int> successfulPairs{0};
    std::atomic<int> failedPairs{0};
    std::atomic<int> skippedPairs{0};

    // Timing
    QDateTime startTime;
    QDateTime lastUpdateTime;
    std::atomic<double> totalProcessingTimeMs{0.0};
    std::atomic<double> averageProcessingTimeMs{0.0};

    // Memory and performance
    std::atomic<size_t> totalOutputSizeBytes{0};
    std::atomic<size_t> peakMemoryUsageBytes{0};
    std::atomic<int> totalPointCloudPoints{0};

    // Current operation
    QString currentOperation;
    QString currentPairName;
    std::atomic<double> currentProgress{0.0}; // 0.0 to 1.0

    // Estimates
    std::atomic<double> estimatedTimeRemainingMs{0.0};
    std::atomic<double> processingRatePerSecond{0.0};

    // Error tracking
    QStringList errorMessages;
    mutable QMutex errorMutex;

    // Delete copy constructor and assignment operator
    BatchProcessingProgress() = default;
    BatchProcessingProgress(const BatchProcessingProgress&) = delete;
    BatchProcessingProgress& operator=(const BatchProcessingProgress&) = delete;
};

/**
 * Copyable progress snapshot for returning to callers
 */
struct BatchProgressSnapshot {
    // Counters
    int totalPairs = 0;
    int processedPairs = 0;
    int successfulPairs = 0;
    int failedPairs = 0;
    int skippedPairs = 0;

    // Timing
    QDateTime startTime;
    QDateTime lastUpdateTime;
    double totalProcessingTimeMs = 0.0;
    double averageProcessingTimeMs = 0.0;

    // Memory and performance
    size_t totalOutputSizeBytes = 0;
    size_t peakMemoryUsageBytes = 0;
    int totalPointCloudPoints = 0;

    // Current operation
    QString currentOperation;
    QString currentPairName;
    double currentProgress = 0.0; // 0.0 to 1.0

    // Estimates
    // Estimates
    std::atomic<double> estimatedTimeRemainingMs{0.0};
    std::atomic<double> processingRatePerSecond{0.0};

    // Error tracking
    QStringList errorMessages;
    mutable QMutex errorMutex;

    // Delete copy constructor and assignment operator
    BatchProcessingProgress() = default;
    BatchProcessingProgress(const BatchProcessingProgress&) = delete;
    BatchProcessingProgress& operator=(const BatchProcessingProgress&) = delete;

    void reset() {
        totalPairs = 0;
        processedPairs = 0;
        successfulPairs = 0;
        failedPairs = 0;
        skippedPairs = 0;
        totalProcessingTimeMs = 0.0;
        averageProcessingTimeMs = 0.0;
        totalOutputSizeBytes = 0;
        peakMemoryUsageBytes = 0;
        totalPointCloudPoints = 0;
        currentProgress = 0.0;
        estimatedTimeRemainingMs = 0.0;
        processingRatePerSecond = 0.0;
        currentOperation.clear();
        currentPairName.clear();

        QMutexLocker locker(&errorMutex);
        errorMessages.clear();
    }
};

/**
 * Copyable progress snapshot for returning to callers
 */
struct BatchProgressSnapshot {
    // Counters
    int totalPairs = 0;
    int processedPairs = 0;
    int successfulPairs = 0;
    int failedPairs = 0;
    int skippedPairs = 0;

    // Timing
    QDateTime startTime;
    QDateTime lastUpdateTime;
    double totalProcessingTimeMs = 0.0;
    double averageProcessingTimeMs = 0.0;

    // Memory and performance
    size_t totalOutputSizeBytes = 0;
    size_t peakMemoryUsageBytes = 0;
    int totalPointCloudPoints = 0;

    // Current operation
    QString currentOperation;
    QString currentPairName;
    double currentProgress = 0.0; // 0.0 to 1.0

    // Estimates
    double estimatedTimeRemainingMs = 0.0;
    double processingRatePerSecond = 0.0;

    // Error tracking
    QStringList errorMessages;
};

/**
 * Main batch processing engine
 * Handles directory scanning, job management, and processing coordination
 */
class BatchProcessor : public QObject {
    Q_OBJECT

public:
    explicit BatchProcessor(QObject* parent = nullptr);
    ~BatchProcessor();

    // Configuration
    void setConfig(const BatchProcessingConfig& config);
    BatchProcessingConfig getConfig() const;

    // Directory operations
    bool scanDirectory(const QString& directoryPath);
    QList<StereoImagePair> discoverStereoPairs(const QString& directoryPath);
    static QStringList getSupportedImageExtensions();

    // Batch processing control
    bool startProcessing();
    void pauseProcessing();
    void resumeProcessing();
    void stopProcessing();
    bool isProcessing() const;

    // Progress and status
    BatchProcessingProgress getProgress() const;
    QList<StereoImagePair> getProcessedPairs() const;
    QList<StereoImagePair> getPendingPairs() const;
    QList<StereoImagePair> getFailedPairs() const;

    // Resume functionality
    bool saveProgressToFile(const QString& filePath) const;
    bool loadProgressFromFile(const QString& filePath);
    QString getDefaultProgressFilePath() const;

    // Report generation
    bool generateProcessingReport(const QString& outputPath) const;
    QJsonObject getProcessingStatistics() const;

signals:
    void progressUpdated(const BatchProcessingProgress& progress);
    void pairStarted(const StereoImagePair& pair);
    void pairCompleted(const StereoImagePair& pair);
    void pairFailed(const StereoImagePair& pair, const QString& error);
    void processingFinished();
    void processingError(const QString& error);

private slots:
    void updateProgress();
    void handleWorkerFinished();

private:
    // Internal processing methods
    void initializeProcessing();
    void cleanupProcessing();
    void processNextBatch();
    bool processSinglePair(StereoImagePair& pair);

    // Validation and quality control
    bool validateStereoPair(const StereoImagePair& pair);
    bool validateOutputQuality(const StereoImagePair& pair, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud);

    // File management
    QString generateOutputFileName(const StereoImagePair& pair) const;
    bool ensureOutputDirectory() const;

    // Memory management
    void checkMemoryUsage();
    size_t getCurrentMemoryUsage() const;

    // Error handling
    void handleProcessingError(StereoImagePair& pair, const QString& error);
    void addErrorMessage(const QString& error) const;

private:
    BatchProcessingConfig m_config;
    BatchProcessingProgress m_progress;

    QList<StereoImagePair> m_stereoPairs;
    mutable QMutex m_pairsMutex;

    std::unique_ptr<stereo_vision::StereoMatcher> m_stereoMatcher;
    std::unique_ptr<stereo_vision::CameraCalibration> m_calibration;
    std::unique_ptr<stereo_vision::PointCloudProcessor> m_pointCloudProcessor;

    // Threading and concurrency
    QThread* m_workerThread;
    std::atomic<bool> m_isProcessing{false};
    std::atomic<bool> m_shouldStop{false};
    std::atomic<bool> m_isPaused{false};

    // Progress tracking
    QTimer* m_progressTimer;
    QMutex m_progressMutex;

    // Resume functionality
    QString m_progressFilePath;
    QTimer* m_autoSaveTimer;
};

} // namespace stereo_vision::batch
