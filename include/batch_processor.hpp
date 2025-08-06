#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDir>
#include <QThread>
#include <QMutex>
#include <QJsonObject>
#include <QJsonArray>
#include <QProgressDialog>
#include <QtConcurrent/QtConcurrent>
#include <QVariant>
#include <QMap>
#include <atomic>
#include <memory>
#include <vector>

#include "stereo_matcher.hpp"
#include "camera_calibration.hpp"
#include "point_cloud_processor.hpp"

namespace stereo_vision {

/**
 * Represents a stereo image pair with processing status
 */
struct StereoImagePair {
    enum Status {
        Pending,
        Processing,
        Completed,
        Failed,
        Skipped
    };

    QString leftImagePath;
    QString rightImagePath;
    QString outputBaseName;
    Status status = Pending;
    QDateTime timestamp;
    QString errorMessage;

    // Processing results
    cv::Size imageSize;
    int pointCloudPoints = 0;
    double processingTimeMs = 0.0;
    size_t outputFileSizeBytes = 0;
};

/**
 * Configuration for batch processing
 */
struct BatchProcessingConfig {
    QString inputDirectory;
    QString outputDirectory;
    QString outputFormat = "PLY"; // PLY or PCD
    QString namingPattern = "{basename}_{timestamp}";

    bool overwriteExisting = false;
    bool validateInputs = true;
    bool enableQualityCheck = true;
    bool generateReport = true;

    double minPointCloudDensity = 0.1;
    int maxMemoryUsageMB = 2048;
    int saveProgressEveryN = 10;

    // Stereo processing parameters
    struct CalibrationParams {
        cv::Mat Q; // Disparity-to-depth mapping matrix
        // Add other calibration parameters as needed
    } calibrationParams;
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
    mutable QStringList errorMessages;
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
    static QStringList getSupportedImageExtensions();
    bool scanDirectory(const QString& directoryPath);

    // Processing control
    bool startProcessing();
    void pauseProcessing();
    void resumeProcessing();
    void stopProcessing();
    bool isProcessing() const;

    // Progress and results
    BatchProgressSnapshot getProgressSnapshot() const;
    QList<StereoImagePair> getProcessedPairs() const;
    QList<StereoImagePair> getPendingPairs() const;
    QList<StereoImagePair> getFailedPairs() const;

    // Persistence
    bool saveProgressToFile(const QString& filePath) const;
    bool loadProgressFromFile(const QString& filePath);
    QString getDefaultProgressFilePath() const;

    // Reporting
    bool generateProcessingReport(const QString& outputPath) const;
    QJsonObject getProcessingStatistics() const;

signals:
    void progressUpdated(const BatchProcessingProgress& progress);
    void pairStarted(const StereoImagePair& pair);
    void pairCompleted(const StereoImagePair& pair);
    void pairFailed(const StereoImagePair& pair, const QString& error);
    void processingFinished();

private slots:
    void updateProgress();
    void handleWorkerFinished();

private:
    // Core processing methods
    void processNextBatch();
    bool processSinglePair(StereoImagePair& pair);
    void initializeProcessing();
    void cleanupProcessing();

    // Discovery and validation
    QList<StereoImagePair> discoverStereoPairs(const QString& directoryPath);
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
    QTimer* m_autoSaveTimer;
    QString m_progressFilePath;
};

} // namespace stereo_vision
