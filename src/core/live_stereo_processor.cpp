#include "live_stereo_processor.hpp"
#include <QDebug>
#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace stereo_vision {

LiveStereoProcessor::LiveStereoProcessor(QObject *parent)
    : QObject(parent), m_isProcessing(false), m_hasCalibration(false),
      m_processingTimer(new QTimer(this)), m_processingFPS(0.0) {
  // Set default settings
  m_settings = ProcessingSettings{};

  // Setup processing timer
  m_processingTimer->setSingleShot(false);
  connect(m_processingTimer, &QTimer::timeout, this,
          &LiveStereoProcessor::processNextFrame);

  initializeProcessors();
}

LiveStereoProcessor::~LiveStereoProcessor() { stopProcessing(); }

void LiveStereoProcessor::setSettings(const ProcessingSettings &settings) {
  m_settings = settings;
  m_processingTimer->setInterval(m_settings.processingInterval);

  // Update processor settings if needed
  if (m_stereoMatcher) {
    // Update stereo matcher parameters
  }
}

void LiveStereoProcessor::setCalibration(
    const CameraCalibration::StereoParameters &params) {
  std::lock_guard<std::mutex> lock(m_queueMutex);
  m_stereoParams = params;
  m_hasCalibration = true;

  qDebug() << "Calibration parameters set for live processing";
}

void LiveStereoProcessor::startProcessing() {
  if (m_isProcessing) {
    return;
  }

  if (!m_hasCalibration) {
    emit processingError("No calibration parameters available");
    return;
  }

  m_isProcessing = true;
  m_processingTimer->start();

  qDebug() << "Live stereo processing started";
}

void LiveStereoProcessor::stopProcessing() {
  if (!m_isProcessing) {
    return;
  }

  m_isProcessing = false;
  m_processingTimer->stop();

  // Clear frame queue
  std::lock_guard<std::mutex> lock(m_queueMutex);
  m_frameQueue.clear();

  qDebug() << "Live stereo processing stopped";
}

void LiveStereoProcessor::processFramePair(const cv::Mat &leftFrame,
                                           const cv::Mat &rightFrame) {
  if (!m_isProcessing || leftFrame.empty() || rightFrame.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(m_queueMutex);

  // Add frame pair to queue
  FramePair framePair;
  framePair.leftFrame = leftFrame.clone();
  framePair.rightFrame = rightFrame.clone();
  framePair.timestamp = std::chrono::high_resolution_clock::now();

  m_frameQueue.push_back(framePair);

  // Limit queue size to prevent memory issues
  const size_t maxQueueSize = 5;
  while (m_frameQueue.size() > maxQueueSize) {
    m_frameQueue.pop_front();
  }
}

cv::Mat LiveStereoProcessor::getLastDisparityMap() {
  std::lock_guard<std::mutex> lock(m_resultsMutex);
  return m_lastDisparityMap.clone();
}

cv::Mat LiveStereoProcessor::getLastPointCloud() {
  std::lock_guard<std::mutex> lock(m_resultsMutex);
  return m_lastPointCloud.clone();
}

cv::Mat LiveStereoProcessor::getLastColoredPointCloud() {
  std::lock_guard<std::mutex> lock(m_resultsMutex);
  return m_lastColoredPointCloud.clone();
}

void LiveStereoProcessor::processNextFrame() {
  if (!m_isProcessing) {
    return;
  }

  FramePair framePair;

  // Get next frame from queue
  {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    if (m_frameQueue.empty()) {
      return;
    }

    framePair = m_frameQueue.front();
    m_frameQueue.pop_front();
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  try {
    // Compute disparity map
    cv::Mat disparityMap;
    if (m_settings.enableDisparityMap) {
      disparityMap =
          computeDisparityMap(framePair.leftFrame, framePair.rightFrame);

      if (!disparityMap.empty()) {
        std::lock_guard<std::mutex> lock(m_resultsMutex);
        m_lastDisparityMap = disparityMap.clone();
        emit disparityMapReady(disparityMap);
      }
    }

    // Generate point cloud
    if (m_settings.enablePointCloud && !disparityMap.empty()) {
      cv::Mat pointCloud =
          generatePointCloud(disparityMap, framePair.leftFrame);

      if (!pointCloud.empty()) {
        std::lock_guard<std::mutex> lock(m_resultsMutex);
        m_lastPointCloud = pointCloud.clone();
        m_lastColoredPointCloud =
            framePair.leftFrame.clone(); // Use left frame as color

        emit pointCloudReady(pointCloud, framePair.leftFrame);
      }
    }

  } catch (const std::exception &e) {
    emit processingError(QString("Processing error: %1").arg(e.what()));
  }

  // Update performance metrics
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);

  // Check if processing took too long
  if (duration.count() > m_settings.maxProcessingTime) {
    qDebug() << "Processing took" << duration.count()
             << "ms (limit:" << m_settings.maxProcessingTime << "ms)";
  }

  updatePerformanceMetrics();
}

void LiveStereoProcessor::initializeProcessors() {
  try {
    m_stereoMatcher = std::make_shared<StereoMatcher>();
    m_pointCloudProcessor = std::make_shared<PointCloudProcessor>();

    qDebug() << "Live stereo processors initialized";
  } catch (const std::exception &e) {
    qDebug() << "Failed to initialize processors:" << e.what();
  }
}

cv::Mat LiveStereoProcessor::computeDisparityMap(const cv::Mat &leftFrame,
                                                 const cv::Mat &rightFrame) {
  if (!m_stereoMatcher || !m_hasCalibration) {
    return cv::Mat();
  }

  // Convert to grayscale if needed
  cv::Mat leftGray, rightGray;
  if (leftFrame.channels() == 3) {
    cv::cvtColor(leftFrame, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightFrame, rightGray, cv::COLOR_BGR2GRAY);
  } else {
    leftGray = leftFrame;
    rightGray = rightFrame;
  }

  // Rectify images using calibration parameters
  cv::Mat leftRectified, rightRectified;

  if (!m_stereoParams.R1.empty() && !m_stereoParams.R2.empty()) {
    cv::Mat map1Left, map2Left, map1Right, map2Right;

    cv::initUndistortRectifyMap(m_stereoParams.left_camera.camera_matrix,
                                m_stereoParams.left_camera.distortion_coeffs,
                                m_stereoParams.R1, m_stereoParams.P1,
                                leftGray.size(), CV_16SC2, map1Left, map2Left);

    cv::initUndistortRectifyMap(
        m_stereoParams.right_camera.camera_matrix,
        m_stereoParams.right_camera.distortion_coeffs, m_stereoParams.R2,
        m_stereoParams.P2, rightGray.size(), CV_16SC2, map1Right, map2Right);

    cv::remap(leftGray, leftRectified, map1Left, map2Left, cv::INTER_LINEAR);
    cv::remap(rightGray, rightRectified, map1Right, map2Right,
              cv::INTER_LINEAR);
  } else {
    leftRectified = leftGray;
    rightRectified = rightGray;
  }

  // Compute disparity using stereo matcher
  cv::Mat disparity;

  if (m_settings.enableGPUAcceleration) {
    // Try GPU acceleration if available
    disparity =
        m_stereoMatcher->computeDisparity(leftRectified, rightRectified);
  }

  if (disparity.empty()) {
    // Fallback to CPU processing
    disparity =
        m_stereoMatcher->computeDisparity(leftRectified, rightRectified);
  }

  // Normalize disparity for visualization
  if (!disparity.empty()) {
    cv::Mat normalizedDisparity;
    disparity.convertTo(normalizedDisparity, CV_8U,
                        255.0 / (m_settings.disparityLevels * 16.0));
    return normalizedDisparity;
  }

  return cv::Mat();
}

cv::Mat LiveStereoProcessor::generatePointCloud(const cv::Mat &disparityMap,
                                                const cv::Mat &leftFrame) {
  if (!m_pointCloudProcessor || !m_hasCalibration || disparityMap.empty()) {
    return cv::Mat();
  }

  // Use the Q matrix from stereo calibration to reproject to 3D
  cv::Mat pointCloud;

  if (!m_stereoParams.Q.empty()) {
    cv::reprojectImageTo3D(disparityMap, pointCloud, m_stereoParams.Q, true);

    // Filter point cloud if enabled
    if (m_settings.enableFiltering && m_pointCloudProcessor) {
      // Apply basic filtering (this would need to be implemented in
      // PointCloudProcessor) For now, just return the raw point cloud
    }

    // Downsample if needed
    if (m_settings.pointCloudDownsample > 0.0) {
      // This would need proper PCL integration for voxel grid filtering
      // For now, just return the point cloud as-is
    }
  }

  return pointCloud;
}

void LiveStereoProcessor::updatePerformanceMetrics() {
  auto currentTime = std::chrono::high_resolution_clock::now();

  if (m_lastProcessTime.time_since_epoch().count() > 0) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        currentTime - m_lastProcessTime);
    double fps = 1000.0 / duration.count();

    m_fpsHistory.push_back(fps);

    // Keep only recent history
    const size_t maxHistory = 10;
    while (m_fpsHistory.size() > maxHistory) {
      m_fpsHistory.pop_front();
    }

    // Calculate average FPS
    double totalFps = 0.0;
    for (double f : m_fpsHistory) {
      totalFps += f;
    }
    m_processingFPS = totalFps / m_fpsHistory.size();

    emit performanceUpdate(m_processingFPS, getQueueSize());
  }

  m_lastProcessTime = currentTime;
}

} // namespace stereo_vision
