#pragma once

#include "camera_calibration.hpp"
#include <QObject>
#include <QTimer>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

namespace stereo_vision {

class AICalibration : public QObject {
  Q_OBJECT

public:
  struct CalibrationSettings {
    cv::Size boardSize{9, 6};     // Checkerboard inner corners
    float squareSize{25.0f};      // Square size in mm
    int requiredFrames{20};       // Minimum frames needed
    int maxFrames{50};            // Maximum frames to collect
    double minDistance{100.0};    // Minimum distance between frames
    double qualityThreshold{0.5}; // Quality threshold for frame acceptance
    bool autoCapture{true};       // Automatic frame capture
    int captureInterval{1000};    // ms between auto captures
  };

  struct CalibrationFrame {
    cv::Mat image;
    std::vector<cv::Point2f> corners;
    double quality;
    cv::Mat pose; // Camera pose for this frame
  };

  explicit AICalibration(QObject *parent = nullptr);
  ~AICalibration();

  // Configuration
  void setSettings(const CalibrationSettings &settings);
  CalibrationSettings getSettings() const { return m_settings; }

  // Frame processing
  bool processFrame(const cv::Mat &leftFrame,
                    const cv::Mat &rightFrame = cv::Mat());
  bool detectChessboard(const cv::Mat &frame, std::vector<cv::Point2f> &corners,
                        double &quality);

  // Calibration management
  void startCalibration();
  void stopCalibration();
  void clearFrames();
  bool isActive() const { return m_active; }

  // Frame management
  int getFrameCount() const { return static_cast<int>(m_leftFrames.size()); }
  int getRequiredFrames() const { return m_settings.requiredFrames; }
  double getProgress() const;

  // Results
  bool hasEnoughFrames() const {
    return getFrameCount() >= m_settings.requiredFrames;
  }
  CameraCalibration::StereoParameters runCalibration();

  // Visualization
  cv::Mat drawCalibrationOverlay(const cv::Mat &frame,
                                 const std::vector<cv::Point2f> &corners,
                                 bool valid);
  std::vector<cv::Mat> getCalibrationVisualization();

signals:
  void frameAccepted(int count, int required);
  void frameRejected(const QString &reason);
  void calibrationProgress(double progress);
  void calibrationComplete(bool success);
  void qualityUpdate(double leftQuality, double rightQuality);

private slots:
  void onAutoCapture();

private:
  bool validateFrame(const CalibrationFrame &frame);
  double calculateFrameQuality(const cv::Mat &frame,
                               const std::vector<cv::Point2f> &corners);
  double calculatePoseDistance(const cv::Mat &pose1, const cv::Mat &pose2);
  cv::Mat estimatePose(const std::vector<cv::Point2f> &corners);

  CalibrationSettings m_settings;
  bool m_active;

  // Frame storage
  std::vector<CalibrationFrame> m_leftFrames;
  std::vector<CalibrationFrame> m_rightFrames;

  // Auto capture
  QTimer *m_autoCaptureTimer;

  // Calibration objects
  std::shared_ptr<CameraCalibration> m_calibration;

  // Quality assessment
  cv::Mat m_lastValidLeftFrame;
  cv::Mat m_lastValidRightFrame;
};

} // namespace stereo_vision
