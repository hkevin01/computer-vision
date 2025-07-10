#pragma once

#include "camera_calibration.hpp"
#include "point_cloud_processor.hpp"
#include "stereo_matcher.hpp"
#include <QObject>
#include <QTimer>
#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace stereo_vision {

class LiveStereoProcessor : public QObject {
  Q_OBJECT

public:
  struct ProcessingSettings {
    bool enableRealTimeProcessing{true};
    bool enableDisparityMap{true};
    bool enablePointCloud{true};
    int processingInterval{100}; // ms between processing cycles
    int maxProcessingTime{50};   // max ms per processing cycle
    bool enableGPUAcceleration{true};
    int disparityLevels{64};
    int blockSize{15};
    double pointCloudDownsample{0.01}; // voxel size for downsampling
    bool enableFiltering{true};
  };

  explicit LiveStereoProcessor(QObject *parent = nullptr);
  ~LiveStereoProcessor();

  // Configuration
  void setSettings(const ProcessingSettings &settings);
  ProcessingSettings getSettings() const { return m_settings; }
  void setCalibration(const CameraCalibration::StereoParameters &params);

  // Processing control
  void startProcessing();
  void stopProcessing();
  bool isProcessing() const { return m_isProcessing; }

  // Frame input
  void processFramePair(const cv::Mat &leftFrame, const cv::Mat &rightFrame);

  // Results access
  cv::Mat getLastDisparityMap();
  cv::Mat getLastPointCloud();
  cv::Mat getLastColoredPointCloud();

  // Performance monitoring
  double getProcessingFPS() const { return m_processingFPS; }
  int getQueueSize() const { return m_frameQueue.size(); }

signals:
  void disparityMapReady(const cv::Mat &disparityMap);
  void pointCloudReady(const cv::Mat &pointCloud, const cv::Mat &colors);
  void processingError(const QString &error);
  void performanceUpdate(double fps, int queueSize);

private slots:
  void processNextFrame();

private:
  struct FramePair {
    cv::Mat leftFrame;
    cv::Mat rightFrame;
    std::chrono::high_resolution_clock::time_point timestamp;
  };

  void initializeProcessors();
  cv::Mat computeDisparityMap(const cv::Mat &leftFrame,
                              const cv::Mat &rightFrame);
  cv::Mat generatePointCloud(const cv::Mat &disparityMap,
                             const cv::Mat &leftFrame);
  void updatePerformanceMetrics();

  ProcessingSettings m_settings;
  std::atomic<bool> m_isProcessing;

  // Calibration
  CameraCalibration::StereoParameters m_stereoParams;
  bool m_hasCalibration;

  // Processors
  std::shared_ptr<StereoMatcher> m_stereoMatcher;
  std::shared_ptr<PointCloudProcessor> m_pointCloudProcessor;

  // Processing pipeline
  QTimer *m_processingTimer;
  std::deque<FramePair> m_frameQueue;
  std::mutex m_queueMutex;

  // Results
  cv::Mat m_lastDisparityMap;
  cv::Mat m_lastPointCloud;
  cv::Mat m_lastColoredPointCloud;
  std::mutex m_resultsMutex;

  // Performance monitoring
  std::chrono::high_resolution_clock::time_point m_lastProcessTime;
  double m_processingFPS;
  std::deque<double> m_fpsHistory;
};

} // namespace stereo_vision
