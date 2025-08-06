#include "batch_processor.hpp"
#include <QApplication>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QJsonArray>
#include <QJsonDocument>
#include <QStandardPaths>
#include <QProcess>
#include <QDebug>
#include <QRegularExpression>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>

namespace stereo_vision {

BatchProcessor::BatchProcessor(QObject* parent)
    : QObject(parent)
    , m_workerThread(nullptr)
    , m_progressTimer(new QTimer(this))
    , m_autoSaveTimer(new QTimer(this))
{
    // Initialize stereo processing components
    m_stereoMatcher = std::make_unique<stereo_vision::StereoMatcher>();
    m_calibration = std::make_unique<stereo_vision::CameraCalibration>();
    m_pointCloudProcessor = std::make_unique<stereo_vision::PointCloudProcessor>();

    // Setup progress tracking
    m_progressTimer->setInterval(1000); // Update every second
    connect(m_progressTimer, &QTimer::timeout, this, &BatchProcessor::updateProgress);

    // Setup auto-save
    m_autoSaveTimer->setInterval(60000); // Auto-save every minute
    connect(m_autoSaveTimer, &QTimer::timeout, [this]() {
        if (m_isProcessing && !m_progressFilePath.isEmpty()) {
            saveProgressToFile(m_progressFilePath);
        }
    });

    // Set default progress file path
    m_progressFilePath = getDefaultProgressFilePath();
}

BatchProcessor::~BatchProcessor() {
    stopProcessing();
}

void BatchProcessor::setConfig(const BatchProcessingConfig& config) {
    if (m_isProcessing) {
        qWarning() << "Cannot change configuration while processing";
        return;
    }

    m_config = config;

    // Initialize stereo matcher with new parameters
    if (m_stereoMatcher) {
        // Convert config parameters to stereo matcher format
        // This would need to be implemented based on actual StereoMatcher API
    }
}

BatchProcessingConfig BatchProcessor::getConfig() const {
    return m_config;
}

QStringList BatchProcessor::getSupportedImageExtensions() {
    return {
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif",
        "*.ppm", "*.pgm", "*.pbm", "*.webp", "*.jp2"
    };
}

bool BatchProcessor::scanDirectory(const QString& directoryPath) {
    qDebug() << "Scanning directory:" << directoryPath;

    QDir dir(directoryPath);
    if (!dir.exists()) {
        addErrorMessage(QString("Directory does not exist: %1").arg(directoryPath));
        return false;
    }

    m_progress.reset();
    m_progress.currentOperation = "Scanning directory...";

    // Discover stereo pairs
    auto discoveredPairs = discoverStereoPairs(directoryPath);

    QMutexLocker locker(&m_pairsMutex);
    m_stereoPairs = discoveredPairs;
    m_progress.totalPairs = m_stereoPairs.size();

    qDebug() << "Found" << m_stereoPairs.size() << "stereo pairs";

    return !m_stereoPairs.isEmpty();
}

QList<StereoImagePair> BatchProcessor::discoverStereoPairs(const QString& directoryPath) {
    QList<StereoImagePair> pairs;
    QDir dir(directoryPath);

    // Get all image files
    QStringList nameFilters = getSupportedImageExtensions();
    QFileInfoList imageFiles = dir.entryInfoList(nameFilters, QDir::Files, QDir::Name);

    // Group files by base name to find stereo pairs
    QMap<QString, QStringList> fileGroups;

    for (const QFileInfo& fileInfo : imageFiles) {
        QString baseName = fileInfo.completeBaseName().toLower();

        // Remove common stereo suffixes to group pairs
        QStringList stereoSuffixes = {
            "_left", "_right", "_l", "_r",
            "left", "right", "L", "R",
            "_0", "_1"
        };

        QString groupKey = baseName;
        for (const QString& suffix : stereoSuffixes) {
            if (baseName.endsWith(suffix.toLower())) {
                groupKey = baseName.left(baseName.length() - suffix.length());
                break;
            }
        }

        fileGroups[groupKey].append(fileInfo.absoluteFilePath());
    }

    // Create stereo pairs from groups with exactly 2 files
    for (auto it = fileGroups.begin(); it != fileGroups.end(); ++it) {
        const QString& groupKey = it.key();
        const QStringList& files = it.value();

        if (files.size() == 2) {
            StereoImagePair pair;

            // Determine which is left and which is right
            QString file1 = files[0];
            QString file2 = files[1];
            QString base1 = QFileInfo(file1).completeBaseName().toLower();
            QString base2 = QFileInfo(file2).completeBaseName().toLower();

            // Simple heuristic: left comes before right alphabetically
            // or contains "left"/"l" vs "right"/"r"
            bool file1IsLeft = false;
            if (base1.contains("left") || base1.contains("_l") || base1.endsWith("_0")) {
                file1IsLeft = true;
            } else if (base2.contains("left") || base2.contains("_l") || base2.endsWith("_0")) {
                file1IsLeft = false;
            } else {
                // Fallback to alphabetical order
                file1IsLeft = (file1 < file2);
            }

            pair.leftImagePath = file1IsLeft ? file1 : file2;
            pair.rightImagePath = file1IsLeft ? file2 : file1;
            pair.outputBaseName = groupKey;
            pair.timestamp = QDateTime::currentDateTime();

            // Validate the pair
            if (validateStereoPair(pair)) {
                pairs.append(pair);
            }
        }
    }

    // Sort pairs by name for consistent processing order
    std::sort(pairs.begin(), pairs.end(), [](const StereoImagePair& a, const StereoImagePair& b) {
        return a.outputBaseName < b.outputBaseName;
    });

    return pairs;
}

bool BatchProcessor::validateStereoPair(const StereoImagePair& pair) {
    if (!m_config.validateInputs) {
        return true; // Skip validation if disabled
    }

    // Check if files exist
    if (!QFile::exists(pair.leftImagePath) || !QFile::exists(pair.rightImagePath)) {
        return false;
    }

    // Load images to check dimensions
    cv::Mat leftImg = cv::imread(pair.leftImagePath.toStdString(), cv::IMREAD_GRAYSCALE);
    cv::Mat rightImg = cv::imread(pair.rightImagePath.toStdString(), cv::IMREAD_GRAYSCALE);

    if (leftImg.empty() || rightImg.empty()) {
        return false;
    }

    // Check if dimensions match
    if (leftImg.size() != rightImg.size()) {
        qWarning() << "Image size mismatch for pair:" << pair.outputBaseName
                   << "Left:" << leftImg.cols << "x" << leftImg.rows
                   << "Right:" << rightImg.cols << "x" << rightImg.rows;
        return false;
    }

    // Check minimum size requirements
    if (leftImg.cols < 100 || leftImg.rows < 100) {
        qWarning() << "Images too small for reliable stereo processing:" << pair.outputBaseName;
        return false;
    }

    return true;
}

bool BatchProcessor::startProcessing() {
    if (m_isProcessing) {
        qWarning() << "Batch processing already in progress";
        return false;
    }

    if (m_stereoPairs.isEmpty()) {
        addErrorMessage("No stereo pairs to process");
        return false;
    }

    qDebug() << "Starting batch processing of" << m_stereoPairs.size() << "pairs";

    m_isProcessing = true;
    m_shouldStop = false;
    m_isPaused = false;
    m_progress.startTime = QDateTime::currentDateTime();
    m_progress.lastUpdateTime = QDateTime::currentDateTime();

    // Ensure output directory exists
    if (!ensureOutputDirectory()) {
        m_isProcessing = false;
        return false;
    }

    // Initialize processing components
    initializeProcessing();

    // Start progress tracking
    m_progressTimer->start();
    m_autoSaveTimer->start();

    // Start processing in separate thread
    m_workerThread = QThread::create([this]() {
        processNextBatch();
    });

    connect(m_workerThread, &QThread::finished, this, &BatchProcessor::handleWorkerFinished);
    m_workerThread->start();

    return true;
}

void BatchProcessor::pauseProcessing() {
    m_isPaused = true;
    m_progress.currentOperation = "Paused";
}

void BatchProcessor::resumeProcessing() {
    m_isPaused = false;
    m_progress.currentOperation = "Processing...";
}

void BatchProcessor::stopProcessing() {
    if (!m_isProcessing) {
        return;
    }

    qDebug() << "Stopping batch processing...";

    m_shouldStop = true;
    m_isPaused = false;

    // Wait for worker thread to finish
    if (m_workerThread && m_workerThread->isRunning()) {
        m_workerThread->quit();
        if (!m_workerThread->wait(5000)) {
            m_workerThread->terminate();
            m_workerThread->wait();
        }
    }

    cleanupProcessing();
}

bool BatchProcessor::isProcessing() const {
    return m_isProcessing;
}

void BatchProcessor::processNextBatch() {
    qDebug() << "Starting batch processing thread";

    QMutexLocker locker(&m_pairsMutex);

    for (int i = 0; i < m_stereoPairs.size() && !m_shouldStop; ++i) {
        // Handle pause
        while (m_isPaused && !m_shouldStop) {
            QThread::msleep(100);
        }

        if (m_shouldStop) break;

        StereoImagePair& pair = m_stereoPairs[i];

        // Skip already processed pairs (for resume functionality)
        if (pair.status == StereoImagePair::Completed ||
            pair.status == StereoImagePair::Skipped) {
            continue;
        }

        // Check memory usage
        checkMemoryUsage();

        m_progress.currentPairName = pair.outputBaseName;
        m_progress.currentOperation = QString("Processing %1/%2: %3")
                                     .arg(i + 1)
                                     .arg(m_stereoPairs.size())
                                     .arg(pair.outputBaseName);

        emit pairStarted(pair);

        // Process the pair
        bool success = processSinglePair(pair);

        if (success) {
            pair.status = StereoImagePair::Completed;
            m_progress.successfulPairs++;
            emit pairCompleted(pair);
        } else {
            pair.status = StereoImagePair::Failed;
            m_progress.failedPairs++;
            emit pairFailed(pair, pair.errorMessage);
        }

        m_progress.processedPairs++;

        // Auto-save progress periodically
        if (m_progress.processedPairs % m_config.saveProgressEveryN == 0) {
            saveProgressToFile(m_progressFilePath);
        }
    }

    qDebug() << "Batch processing thread completed";
}

bool BatchProcessor::processSinglePair(StereoImagePair& pair) {
    auto startTime = std::chrono::high_resolution_clock::now();
    pair.status = StereoImagePair::Processing;

    try {
        // Load images
        cv::Mat leftImg = cv::imread(pair.leftImagePath.toStdString(), cv::IMREAD_COLOR);
        cv::Mat rightImg = cv::imread(pair.rightImagePath.toStdString(), cv::IMREAD_COLOR);

        if (leftImg.empty() || rightImg.empty()) {
            pair.errorMessage = "Failed to load stereo images";
            return false;
        }

        pair.imageSize = leftImg.size();

        // Compute disparity
        cv::Mat disparity = m_stereoMatcher->computeDisparity(leftImg, rightImg);
        if (disparity.empty()) {
            pair.errorMessage = "Failed to compute disparity map";
            return false;
        }

        // Generate point cloud
        auto pointCloud = m_stereoMatcher->generatePointCloud(disparity, leftImg);
        if (!pointCloud || pointCloud->empty()) {
            pair.errorMessage = "Failed to generate point cloud";
            return false;
        }

        // Quality check
        if (m_config.enableQualityCheck && !validateOutputQuality(pair, pointCloud)) {
            pair.errorMessage = "Point cloud quality below threshold";
            return false;
        }

        pair.pointCloudPoints = pointCloud->size();
        m_progress.totalPointCloudPoints.fetch_add(pair.pointCloudPoints);

        // Process point cloud (filtering, etc.)
        if (m_pointCloudProcessor) {
            pointCloud = m_pointCloudProcessor->filterPointCloud(pointCloud);
        }

        // Generate output filename
        QString outputPath = generateOutputFileName(pair);

        // Check if output already exists
        if (QFile::exists(outputPath) && !m_config.overwriteExisting) {
            pair.status = StereoImagePair::Skipped;
            pair.errorMessage = "Output file exists and overwrite disabled";
            m_progress.skippedPairs++;
            return true; // Not an error, just skipped
        }

        // Save point cloud
        bool saveSuccess = false;
        if (m_config.outputFormat.toLower() == "ply") {
            saveSuccess = (pcl::io::savePLYFileBinary(outputPath.toStdString(), *pointCloud) == 0);
        } else if (m_config.outputFormat.toLower() == "pcd") {
            saveSuccess = (pcl::io::savePCDFileBinary(outputPath.toStdString(), *pointCloud) == 0);
        } else {
            pair.errorMessage = "Unsupported output format: " + m_config.outputFormat;
            return false;
        }

        if (!saveSuccess) {
            pair.errorMessage = "Failed to save point cloud to: " + outputPath;
            return false;
        }

        // Get output file size
        QFileInfo outputFileInfo(outputPath);
        pair.outputFileSizeBytes = outputFileInfo.size();
        m_progress.totalOutputSizeBytes.fetch_add(pair.outputFileSizeBytes);

        // Calculate processing time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        pair.processingTimeMs = duration.count();

        // Update total processing time (atomic double needs special handling)
        double oldTime = m_progress.totalProcessingTimeMs.load();
        while (!m_progress.totalProcessingTimeMs.compare_exchange_weak(
            oldTime, oldTime + pair.processingTimeMs)) {
            // retry if concurrent modification
        }

        // Update average processing time
        if (m_progress.processedPairs > 0) {
            m_progress.averageProcessingTimeMs = m_progress.totalProcessingTimeMs.load() / m_progress.processedPairs.load();
        }

        return true;

    } catch (const std::exception& e) {
        pair.errorMessage = QString("Exception during processing: %1").arg(e.what());
        return false;
    } catch (...) {
        pair.errorMessage = "Unknown exception during processing";
        return false;
    }
}

bool BatchProcessor::validateOutputQuality(const StereoImagePair& pair,
                                          const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud) {
    if (!pointCloud || pointCloud->empty()) {
        return false;
    }

    // Calculate point density (points per pixel)
    double totalPixels = pair.imageSize.width * pair.imageSize.height;
    double density = static_cast<double>(pointCloud->size()) / totalPixels;

    if (density < m_config.minPointCloudDensity) {
        qDebug() << "Point cloud density too low:" << density << "< threshold:" << m_config.minPointCloudDensity;
        return false;
    }

    // Check for valid 3D coordinates (not all NaN/inf)
    int validPoints = 0;
    for (const auto& point : *pointCloud) {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            validPoints++;
        }
    }

    double validRatio = static_cast<double>(validPoints) / pointCloud->size();
    if (validRatio < 0.5) { // At least 50% valid points
        qDebug() << "Too many invalid points in cloud:" << validRatio;
        return false;
    }

    return true;
}

QString BatchProcessor::generateOutputFileName(const StereoImagePair& pair) const {
    QString pattern = m_config.namingPattern;
    QString outputName = pattern;

    // Replace placeholders
    outputName.replace("{basename}", pair.outputBaseName);
    outputName.replace("{timestamp}", pair.timestamp.toString("yyyyMMdd_hhmmss"));
    outputName.replace("{date}", pair.timestamp.toString("yyyyMMdd"));
    outputName.replace("{time}", pair.timestamp.toString("hhmmss"));

    // Add extension
    outputName += "." + m_config.outputFormat.toLower();

    return QDir(m_config.outputDirectory).absoluteFilePath(outputName);
}

bool BatchProcessor::ensureOutputDirectory() const {
    QDir dir(m_config.outputDirectory);
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {
            addErrorMessage(QString("Failed to create output directory: %1").arg(m_config.outputDirectory));
            return false;
        }
    }
    return true;
}

void BatchProcessor::checkMemoryUsage() {
    size_t currentMemory = getCurrentMemoryUsage();
    if (currentMemory > m_progress.peakMemoryUsageBytes) {
        m_progress.peakMemoryUsageBytes = currentMemory;
    }

    // If memory usage is too high, trigger garbage collection or wait
    size_t maxMemoryBytes = static_cast<size_t>(m_config.maxMemoryUsageMB) * 1024 * 1024;
    if (currentMemory > maxMemoryBytes) {
        qWarning() << "Memory usage high:" << (currentMemory / 1024 / 1024) << "MB";
        // Could implement memory pressure handling here
    }
}

size_t BatchProcessor::getCurrentMemoryUsage() const {
    // Platform-specific memory usage detection
    // This is a simplified implementation
#ifdef Q_OS_LINUX
    QFile file("/proc/self/status");
    if (file.open(QIODevice::ReadOnly)) {
        QTextStream stream(&file);
        QString line;
        while (stream.readLineInto(&line)) {
            if (line.startsWith("VmRSS:")) {
                QStringList parts = line.split(QRegularExpression("\\s+"));
                if (parts.size() >= 2) {
                    return parts[1].toLongLong() * 1024; // Convert KB to bytes
                }
            }
        }
    }
#endif
    return 0; // Fallback
}

void BatchProcessor::handleProcessingError(StereoImagePair& pair, const QString& error) {
    pair.errorMessage = error;
    pair.status = StereoImagePair::Failed;
    addErrorMessage(QString("Failed to process %1: %2").arg(pair.outputBaseName, error));
}

void BatchProcessor::addErrorMessage(const QString& error) const {
    QMutexLocker locker(&m_progress.errorMutex);
    m_progress.errorMessages.append(QString("%1: %2")
                                   .arg(QDateTime::currentDateTime().toString("hh:mm:ss"), error));
}

void BatchProcessor::updateProgress() {
    m_progress.lastUpdateTime = QDateTime::currentDateTime();

    // Calculate processing rate
    if (m_progress.processedPairs > 0 && !m_progress.startTime.isNull()) {
        qint64 elapsedMs = m_progress.startTime.msecsTo(m_progress.lastUpdateTime);
        if (elapsedMs > 0) {
            m_progress.processingRatePerSecond = (m_progress.processedPairs * 1000.0) / elapsedMs;
        }
    }

    // Estimate time remaining
    if (m_progress.processingRatePerSecond > 0) {
        int remainingPairs = m_progress.totalPairs - m_progress.processedPairs;
        m_progress.estimatedTimeRemainingMs = (remainingPairs / m_progress.processingRatePerSecond) * 1000.0;
    }

    // Calculate overall progress
    if (m_progress.totalPairs > 0) {
        m_progress.currentProgress = static_cast<double>(m_progress.processedPairs) / m_progress.totalPairs;
    }

    emit progressUpdated(m_progress);
}

void BatchProcessor::handleWorkerFinished() {
    qDebug() << "Batch processing worker finished";

    m_progressTimer->stop();
    m_autoSaveTimer->stop();

    // Final progress update
    updateProgress();

    // Save final progress
    saveProgressToFile(m_progressFilePath);

    // Generate report if enabled
    if (m_config.generateReport) {
        QString reportPath = QDir(m_config.outputDirectory).absoluteFilePath("processing_report.json");
        generateProcessingReport(reportPath);
    }

    m_isProcessing = false;
    m_progress.currentOperation = "Completed";

    if (m_workerThread) {
        m_workerThread->deleteLater();
        m_workerThread = nullptr;
    }

    emit processingFinished();
}

void BatchProcessor::initializeProcessing() {
    // Initialize stereo matcher with current parameters
    if (m_stereoMatcher) {
        // This would initialize the matcher with current calibration
        // Implementation depends on actual StereoMatcher API
    }
}

void BatchProcessor::cleanupProcessing() {
    m_progressTimer->stop();
    m_autoSaveTimer->stop();
    m_isProcessing = false;
}

BatchProgressSnapshot BatchProcessor::getProgressSnapshot() const {
    BatchProgressSnapshot snapshot;

    // Copy atomic values
    snapshot.totalPairs = m_progress.totalPairs.load();
    snapshot.processedPairs = m_progress.processedPairs.load();
    snapshot.successfulPairs = m_progress.successfulPairs.load();
    snapshot.failedPairs = m_progress.failedPairs.load();
    snapshot.skippedPairs = m_progress.skippedPairs.load();

    // Copy timing information
    snapshot.startTime = m_progress.startTime;
    snapshot.lastUpdateTime = m_progress.lastUpdateTime;
    snapshot.totalProcessingTimeMs = m_progress.totalProcessingTimeMs.load();
    snapshot.averageProcessingTimeMs = m_progress.averageProcessingTimeMs.load();

    // Copy performance metrics
    snapshot.totalOutputSizeBytes = m_progress.totalOutputSizeBytes.load();
    snapshot.peakMemoryUsageBytes = m_progress.peakMemoryUsageBytes.load();
    snapshot.totalPointCloudPoints = m_progress.totalPointCloudPoints.load();

    // Copy current operation info
    snapshot.currentOperation = m_progress.currentOperation;
    snapshot.currentPairName = m_progress.currentPairName;
    snapshot.currentProgress = m_progress.currentProgress.load();

    // Copy estimates
    snapshot.estimatedTimeRemainingMs = m_progress.estimatedTimeRemainingMs.load();
    snapshot.processingRatePerSecond = m_progress.processingRatePerSecond.load();

    // Copy error messages (with mutex protection)
    {
        QMutexLocker locker(&m_progress.errorMutex);
        snapshot.errorMessages = m_progress.errorMessages;
    }

    return snapshot;
}

QList<StereoImagePair> BatchProcessor::getProcessedPairs() const {
    m_pairsMutex.lock();
    QList<StereoImagePair> processed;
    for (const auto& pair : m_stereoPairs) {
        if (pair.status == StereoImagePair::Completed) {
            processed.append(pair);
        }
    }
    m_pairsMutex.unlock();
    return processed;
}

QList<StereoImagePair> BatchProcessor::getPendingPairs() const {
    m_pairsMutex.lock();
    QList<StereoImagePair> pending;
    for (const auto& pair : m_stereoPairs) {
        if (pair.status == StereoImagePair::Pending) {
            pending.append(pair);
        }
    }
    m_pairsMutex.unlock();
    return pending;
}

QList<StereoImagePair> BatchProcessor::getFailedPairs() const {
    m_pairsMutex.lock();
    QList<StereoImagePair> failed;
    for (const auto& pair : m_stereoPairs) {
        if (pair.status == StereoImagePair::Failed) {
            failed.append(pair);
        }
    }
    m_pairsMutex.unlock();
    return failed;
}

bool BatchProcessor::saveProgressToFile(const QString& filePath) const {
    QJsonObject progressJson;

    // Save configuration
    QJsonObject configJson;
    configJson["inputDirectory"] = m_config.inputDirectory;
    configJson["outputDirectory"] = m_config.outputDirectory;
    configJson["outputFormat"] = m_config.outputFormat;
    configJson["namingPattern"] = m_config.namingPattern;
    progressJson["config"] = configJson;

    // Save pairs status
    QJsonArray pairsJson;
    m_pairsMutex.lock();
    for (const auto& pair : m_stereoPairs) {
        QJsonObject pairJson;
        pairJson["leftImagePath"] = pair.leftImagePath;
        pairJson["rightImagePath"] = pair.rightImagePath;
        pairJson["outputBaseName"] = pair.outputBaseName;
        pairJson["status"] = static_cast<int>(pair.status);
        pairJson["errorMessage"] = pair.errorMessage;
        pairJson["processingTimeMs"] = pair.processingTimeMs;
        pairJson["outputFileSizeBytes"] = static_cast<qint64>(pair.outputFileSizeBytes);
        pairJson["pointCloudPoints"] = pair.pointCloudPoints;
        pairsJson.append(pairJson);
    }
    m_pairsMutex.unlock();
    progressJson["pairs"] = pairsJson;    // Save progress statistics
    QJsonObject statsJson;
    statsJson["totalPairs"] = m_progress.totalPairs.load();
    statsJson["processedPairs"] = m_progress.processedPairs.load();
    statsJson["successfulPairs"] = m_progress.successfulPairs.load();
    statsJson["failedPairs"] = m_progress.failedPairs.load();
    statsJson["skippedPairs"] = m_progress.skippedPairs.load();
    statsJson["startTime"] = m_progress.startTime.toString(Qt::ISODate);
    progressJson["statistics"] = statsJson;

    // Write to file
    QJsonDocument doc(progressJson);
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        return true;
    }

    return false;
}

bool BatchProcessor::loadProgressFromFile(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }

    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError) {
        return false;
    }

    QJsonObject progressJson = doc.object();

    // Load pairs
    QJsonArray pairsJson = progressJson["pairs"].toArray();
    QList<StereoImagePair> loadedPairs;

    for (const auto& value : pairsJson) {
        QJsonObject pairJson = value.toObject();
        StereoImagePair pair;
        pair.leftImagePath = pairJson["leftImagePath"].toString();
        pair.rightImagePath = pairJson["rightImagePath"].toString();
        pair.outputBaseName = pairJson["outputBaseName"].toString();
        pair.status = static_cast<StereoImagePair::Status>(pairJson["status"].toInt());
        pair.errorMessage = pairJson["errorMessage"].toString();
        pair.processingTimeMs = pairJson["processingTimeMs"].toDouble();
        pair.outputFileSizeBytes = pairJson["outputFileSizeBytes"].toInt();
        pair.pointCloudPoints = pairJson["pointCloudPoints"].toInt();
        loadedPairs.append(pair);
    }

    m_pairsMutex.lock();
    m_stereoPairs = loadedPairs;
    m_pairsMutex.unlock();

    // Load statistics
    QJsonObject statsJson = progressJson["statistics"].toObject();
    m_progress.totalPairs = statsJson["totalPairs"].toInt();
    m_progress.processedPairs = statsJson["processedPairs"].toInt();
    m_progress.successfulPairs = statsJson["successfulPairs"].toInt();
    m_progress.failedPairs = statsJson["failedPairs"].toInt();
    m_progress.skippedPairs = statsJson["skippedPairs"].toInt();

    return true;
}

QString BatchProcessor::getDefaultProgressFilePath() const {
    QString dataDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dataDir);
    return QDir(dataDir).absoluteFilePath("batch_progress.json");
}

bool BatchProcessor::generateProcessingReport(const QString& outputPath) const {
    QJsonObject report = getProcessingStatistics();

    QJsonDocument doc(report);
    QFile file(outputPath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        return true;
    }

    return false;
}

QJsonObject BatchProcessor::getProcessingStatistics() const {
    QJsonObject stats;

    // Basic statistics
    stats["totalPairs"] = m_progress.totalPairs.load();
    stats["processedPairs"] = m_progress.processedPairs.load();
    stats["successfulPairs"] = m_progress.successfulPairs.load();
    stats["failedPairs"] = m_progress.failedPairs.load();
    stats["skippedPairs"] = m_progress.skippedPairs.load();

    // Timing statistics
    stats["startTime"] = m_progress.startTime.toString(Qt::ISODate);
    stats["totalProcessingTimeMs"] = m_progress.totalProcessingTimeMs.load();
    stats["averageProcessingTimeMs"] = m_progress.averageProcessingTimeMs.load();
    stats["processingRatePerSecond"] = m_progress.processingRatePerSecond.load();

    // Output statistics
    stats["totalOutputSizeBytes"] = static_cast<qint64>(m_progress.totalOutputSizeBytes.load());
    stats["totalPointCloudPoints"] = m_progress.totalPointCloudPoints.load();
    stats["peakMemoryUsageBytes"] = static_cast<qint64>(m_progress.peakMemoryUsageBytes.load());

    // Success rate
    int total = m_progress.processedPairs.load();
    if (total > 0) {
        stats["successRate"] = static_cast<double>(m_progress.successfulPairs.load()) / total;
    }

    // Error messages
    QJsonArray errorsJson;
    m_progress.errorMutex.lock();
    for (const QString& error : m_progress.errorMessages) {
        errorsJson.append(error);
    }
    m_progress.errorMutex.unlock();
    stats["errors"] = errorsJson;

    return stats;
}

} // namespace stereo_vision
