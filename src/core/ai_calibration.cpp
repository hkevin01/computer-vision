#include "ai_calibration.hpp"
#include <QDebug>
#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace stereo_vision {

AICalibration::AICalibration(QObject *parent)
    : QObject(parent), m_active(false), m_autoCaptureTimer(new QTimer(this)),
      m_calibration(std::make_shared<CameraCalibration>()) {
  // Set default settings
  m_settings = CalibrationSettings{};

  // Setup auto capture timer
  m_autoCaptureTimer->setSingleShot(false);
  connect(m_autoCaptureTimer, &QTimer::timeout, this,
          &AICalibration::onAutoCapture);
}

AICalibration::~AICalibration() { stopCalibration(); }

void AICalibration::setSettings(const CalibrationSettings &settings) {
  m_settings = settings;
  m_autoCaptureTimer->setInterval(m_settings.captureInterval);
}

bool AICalibration::processFrame(const cv::Mat &leftFrame,
                                 const cv::Mat &rightFrame) {
  if (!m_active || leftFrame.empty()) {
    return false;
  }

  // Process left frame
  std::vector<cv::Point2f> leftCorners;
  double leftQuality;
  bool leftValid = detectChessboard(leftFrame, leftCorners, leftQuality);

  // Process right frame (if provided)
  std::vector<cv::Point2f> rightCorners;
  double rightQuality = 0.0;
  bool rightValid = true;

  if (!rightFrame.empty()) {
    rightValid = detectChessboard(rightFrame, rightCorners, rightQuality);
  }

  emit qualityUpdate(leftQuality, rightQuality);

  // For auto capture mode, frames are captured automatically
  if (m_settings.autoCapture) {
    return leftValid && rightValid;
  }

  // Manual capture mode - just validate
  return leftValid && rightValid;
}

bool AICalibration::detectChessboard(const cv::Mat &frame,
                                     std::vector<cv::Point2f> &corners,
                                     double &quality) {
  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame.clone();
  }

  // Detect chessboard corners
  bool found = cv::findChessboardCorners(gray, m_settings.boardSize, corners,
                                         cv::CALIB_CB_ADAPTIVE_THRESH |
                                             cv::CALIB_CB_NORMALIZE_IMAGE |
                                             cv::CALIB_CB_FILTER_QUADS);

  if (found) {
    // Refine corner positions
    cv::cornerSubPix(
        gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                         0.1));

    // Calculate quality
    quality = calculateFrameQuality(gray, corners);

    return quality > m_settings.qualityThreshold;
  }

  quality = 0.0;
  return false;
}

void AICalibration::startCalibration() {
  if (m_active) {
    return;
  }

  m_active = true;
  clearFrames();

  if (m_settings.autoCapture) {
    m_autoCaptureTimer->start();
  }

  qDebug() << "AI Calibration started - collecting" << m_settings.requiredFrames
           << "frames";
}

void AICalibration::stopCalibration() {
  if (!m_active) {
    return;
  }

  m_active = false;
  m_autoCaptureTimer->stop();

  qDebug() << "AI Calibration stopped";
}

void AICalibration::clearFrames() {
  m_leftFrames.clear();
  m_rightFrames.clear();
  emit calibrationProgress(0.0);
}

double AICalibration::getProgress() const {
  if (m_settings.requiredFrames == 0) {
    return 0.0;
  }
  return std::min(1.0, static_cast<double>(getFrameCount()) /
                           m_settings.requiredFrames);
}

CameraCalibration::StereoParameters AICalibration::runCalibration() {
  if (!hasEnoughFrames()) {
    throw std::runtime_error("Not enough calibration frames collected");
  }

  // Prepare object points
  std::vector<std::vector<cv::Point3f>> objectPoints;
  std::vector<cv::Point3f> objp;

  for (int i = 0; i < m_settings.boardSize.height; ++i) {
    for (int j = 0; j < m_settings.boardSize.width; ++j) {
      objp.push_back(
          cv::Point3f(j * m_settings.squareSize, i * m_settings.squareSize, 0));
    }
  }

  // Prepare image points
  std::vector<std::vector<cv::Point2f>> leftImagePoints, rightImagePoints;

  for (size_t i = 0; i < m_leftFrames.size(); ++i) {
    objectPoints.push_back(objp);
    leftImagePoints.push_back(m_leftFrames[i].corners);

    if (i < m_rightFrames.size()) {
      rightImagePoints.push_back(m_rightFrames[i].corners);
    }
  }

  // Get image size
  cv::Size imageSize = m_leftFrames[0].image.size();

  // Perform calibration
  if (rightImagePoints.empty()) {
    // Single camera calibration
    std::vector<cv::Mat> leftImages;
    for (const auto &frame : m_leftFrames) {
      leftImages.push_back(frame.image);
    }

    auto leftParams = m_calibration->calibrateSingleCamera(
        leftImages, m_settings.boardSize, m_settings.squareSize);

    // Create stereo parameters with duplicate left camera
    CameraCalibration::StereoParameters params;
    params.left_camera = leftParams;
    params.right_camera = leftParams;

    return params;
  } else {
    // Stereo calibration
    std::vector<cv::Mat> leftImages, rightImages;
    for (const auto &frame : m_leftFrames) {
      leftImages.push_back(frame.image);
    }
    for (const auto &frame : m_rightFrames) {
      rightImages.push_back(frame.image);
    }

    return m_calibration->calibrateStereoCamera(
        leftImages, rightImages, m_settings.boardSize, m_settings.squareSize);
  }
}

cv::Mat AICalibration::drawCalibrationOverlay(
    const cv::Mat &frame, const std::vector<cv::Point2f> &corners, bool valid) {
  cv::Mat overlay = frame.clone();

  if (!corners.empty()) {
    // Draw chessboard corners
    cv::drawChessboardCorners(overlay, m_settings.boardSize, corners, valid);

    // Draw quality indicator
    cv::Scalar color = valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::circle(overlay, cv::Point(30, 30), 15, color, -1);

    // Draw frame count
    std::string text = std::to_string(getFrameCount()) + "/" +
                       std::to_string(m_settings.requiredFrames);
    cv::putText(overlay, text, cv::Point(60, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                color, 2);
  }

  return overlay;
}

void AICalibration::onAutoCapture() {
  // This will be called when auto-capture conditions are met
  emit frameAccepted(getFrameCount(), m_settings.requiredFrames);
  emit calibrationProgress(getProgress());

  if (hasEnoughFrames()) {
    stopCalibration();
    emit calibrationComplete(true);
  }
}

bool AICalibration::validateFrame(const CalibrationFrame &frame) {
  if (frame.corners.empty() || frame.quality < m_settings.qualityThreshold) {
    return false;
  }

  // Check minimum distance from existing frames
  for (const auto &existingFrame : m_leftFrames) {
    double distance = calculatePoseDistance(frame.pose, existingFrame.pose);
    if (distance < m_settings.minDistance) {
      return false;
    }
  }

  return true;
}

double
AICalibration::calculateFrameQuality(const cv::Mat &frame,
                                     const std::vector<cv::Point2f> &corners) {
  if (corners.empty()) {
    return 0.0;
  }

  // Calculate various quality metrics
  double sharpness = 0.0;
  double coverage = 0.0;
  double uniformity = 0.0;

  // Sharpness: Laplacian variance
  cv::Mat laplacian;
  cv::Laplacian(frame, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  sharpness = stddev[0] * stddev[0];

  // Coverage: How much of the image is covered by the chessboard
  cv::Rect boundingRect = cv::boundingRect(corners);
  double area = boundingRect.area();
  double imageArea = frame.rows * frame.cols;
  coverage = area / imageArea;

  // Uniformity: How evenly distributed the corners are
  // Simple measure: standard deviation of distances from center
  cv::Point2f center(frame.cols / 2.0f, frame.rows / 2.0f);
  std::vector<double> distances;
  for (const auto &corner : corners) {
    double dist = cv::norm(corner - center);
    distances.push_back(dist);
  }

  if (!distances.empty()) {
    double meanDist = std::accumulate(distances.begin(), distances.end(), 0.0) /
                      distances.size();
    double variance = 0.0;
    for (double dist : distances) {
      variance += (dist - meanDist) * (dist - meanDist);
    }
    variance /= distances.size();
    uniformity = 1.0 / (1.0 + variance);
  }

  // Combine metrics (weights can be adjusted)
  double quality = 0.4 * std::tanh(sharpness / 1000.0) +
                   0.3 * std::min(1.0, coverage * 4.0) + 0.3 * uniformity;

  return std::min(1.0, std::max(0.0, quality));
}

double AICalibration::calculatePoseDistance(const cv::Mat &pose1,
                                            const cv::Mat &pose2) {
  if (pose1.empty() || pose2.empty()) {
    return std::numeric_limits<double>::max();
  }

  // Simple Euclidean distance between pose translations
  cv::Mat diff = pose1 - pose2;
  return cv::norm(diff);
}

cv::Mat AICalibration::estimatePose(const std::vector<cv::Point2f> &corners) {
  // Simplified pose estimation - in a real implementation,
  // you would use solvePnP with known object points
  if (corners.empty()) {
    return cv::Mat();
  }

  // For now, just return the centroid as a simple pose representation
  cv::Point2f centroid(0, 0);
  for (const auto &corner : corners) {
    centroid += corner;
  }
  centroid.x /= corners.size();
  centroid.y /= corners.size();

  cv::Mat pose = (cv::Mat_<double>(2, 1) << centroid.x, centroid.y);
  return pose;
}

std::vector<cv::Mat> AICalibration::getCalibrationVisualization() {
  std::vector<cv::Mat> visualizations;

  for (const auto &frame : m_leftFrames) {
    cv::Mat vis = drawCalibrationOverlay(frame.image, frame.corners, true);
    visualizations.push_back(vis);
  }

  return visualizations;
}

} // namespace stereo_vision
