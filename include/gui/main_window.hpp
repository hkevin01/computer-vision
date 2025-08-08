#pragma once

// System includes
#include <memory>
#include <vector>
#include <map>

// Qt includes
#include <QMainWindow>
#include <QWidget>
#include <QString>
#include <QTimer>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QProgressBar>
#include <QLabel>
#include <QSize>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QStandardPaths>
#include <QFileDialog>
#include <QDateTime>
#include <QDir>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QPushButton>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFrame>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Project includes
#include "camera_calibration.hpp"
#include "camera_manager.hpp"
#include "point_cloud_processor.hpp"
#include "stereo_matcher.hpp"
#include "gui/image_display_widget.hpp"
#include "gui/point_cloud_widget.hpp"
#include "gui/batch_processing_window.hpp"
#include "multicam/multi_camera_system_simple.hpp" // new multicam system

namespace stereo_vision::gui {

// Forward declare our own GUI widgets in the stereo_vision::gui namespace
class ImageDisplayWidget;
class ParameterPanel;
class PointCloudWidget;
class EpipolarChecker;

} // namespace stereo_vision::gui

namespace stereo_vision::batch {
class BatchProcessingWindow;
} // namespace stereo_vision::batch

namespace stereo_vision::gui {

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private slots:
  void openLeftImage();
  void openRightImage();
  void openStereoFolder();
  void loadCalibration();
  void saveCalibration();
  void runCalibration();
  void processStereoImages();
  void exportPointCloud();
  void showAbout();
  void showKeyboardShortcuts();
  void onParameterChanged();
  void onProcessingFinished();
  void resetView();
  void showCameraSelector();
  void startWebcamCapture();
  void stopWebcamCapture();
  void captureLeftImage();
  void captureRightImage();
  void captureStereoImage();
  void onFrameReady();
  void onCameraSelectionChanged();
  void startAICalibration();
  void onCalibrationProgress(int progress);
  void onCalibrationComplete();
  void captureCalibrationFrame();
  void toggleLiveProcessing();
  void onLiveFrameProcessed();
  void updateDisparityMap();
  void updatePointCloud();
  void openBatchProcessing();
  void openEpipolarChecker();
  void refreshCameraStatus();
  void toggleProfiling(bool checked = false);
  void updateProfilingStats(); // periodic profiling snapshot
  void updateSyncStatus(); // new: periodic sync stats update
  void performRetryAttempt(); // new: non-blocking retry attempt chain

private:
  void setupUI();
  void setupMenuBar();
  void setupCentralWidget();
  void setupStatusBar();
  void connectSignals();
  void updateUI();
  void initializeCameraSystem();
  void showCameraErrorDialog(const QString &title, const QString &message, const QString &details = QString());
  void logCameraOperation(const QString &operation, bool success, const QString &details = QString());
  void retryCameraConnection(int cameraId, int maxRetries = 3); // refactored to schedule attempts
  void updateCameraStatusIndicators();

  // Central widget components
  QWidget *m_centralWidget;
  QSplitter *m_mainSplitter;
  QTabWidget *m_imageTabWidget;
  ImageDisplayWidget *m_leftImageWidget;
  ImageDisplayWidget *m_rightImageWidget;
  ImageDisplayWidget *m_disparityWidget;
  PointCloudWidget *m_pointCloudWidget;
  ParameterPanel *m_parameterPanel;

  // Menu and toolbar
  QMenuBar *m_menuBar;
  QMenu *m_fileMenu;
  QMenu *m_processMenu;
  QMenu *m_viewMenu;
  QMenu *m_helpMenu;

  QAction *m_openLeftAction;
  QAction *m_openRightAction;
  QAction *m_openFolderAction;
  QAction *m_loadCalibrationAction;
  QAction *m_saveCalibrationAction;
  QAction *m_exitAction;
  QAction *m_calibrateAction;
  QAction *m_processAction;
  QAction *m_batchProcessAction;
  QAction *m_epipolarCheckerAction;
  QAction *m_exportAction;
  QAction *m_aboutAction;
  QAction *m_shortcutsAction;

  // Webcam capture actions
  QAction *m_cameraSelectAction;
  QAction *m_startCaptureAction;
  QAction *m_stopCaptureAction;
  QAction *m_captureLeftAction;
  QAction *m_captureRightAction;
  QAction *m_captureStereoAction;

  // Live processing actions
  QAction *m_liveProcessingAction;
  QAction *m_showDisparityAction;
  QAction *m_showPointCloudAction;
  QAction *m_enableProfilingAction; // toggle performance profiling

  // AI Calibration actions
  QAction *m_aiCalibrationAction;

  // Status bar
  QStatusBar *m_statusBar;
  QProgressBar *m_progressBar;
  QLabel *m_statusLabel;
  QLabel *m_syncStatusLabel; // new: sync stats/quality label

  // Camera status indicators
  QGroupBox *m_cameraStatusGroup;
  QLabel *m_leftCameraStatusLabel;
  QLabel *m_rightCameraStatusLabel;
  QPushButton *m_retryConnectionButton;
  QTextEdit *m_debugLogOutput;

  // Processing components
  std::shared_ptr<stereo_vision::CameraCalibration> m_calibration;
  stereo_vision::CameraCalibration::StereoParameters m_stereoParams;
  std::shared_ptr<stereo_vision::StereoMatcher> m_stereoMatcher;
  std::shared_ptr<stereo_vision::PointCloudProcessor> m_pointCloudProcessor;
  std::shared_ptr<stereo_vision::CameraManager> m_cameraManager;
  std::shared_ptr<stereovision::multicam::MultiCameraSystem> m_multiCameraSystem; // new multicam system

  // Data
  QString m_leftImagePath;
  QString m_rightImagePath;
  QString m_calibrationPath;
  QString m_outputPath;

  // Processing state
  QTimer *m_processingTimer;
  bool m_isProcessing;
  bool m_hasCalibration;
  bool m_hasImages;

  // Webcam capture state
  QTimer *m_captureTimer;
  bool m_isCapturing;
  bool m_leftCameraConnected;
  bool m_rightCameraConnected;
  int m_selectedLeftCamera;
  int m_selectedRightCamera;
  cv::Mat m_lastLeftFrame;
  cv::Mat m_lastRightFrame;

  // Live processing state
  QTimer *m_liveProcessingTimer;
  bool m_liveProcessingEnabled;
  cv::Mat m_lastDisparityMap;
  cv::Mat m_lastPointCloud;

  // Profiling state
  QTimer *m_profilingTimer; // periodic profiler snapshot timer

  // Sync status update timer (reuse profiling if desired but separate for clarity)
  QTimer *m_syncUpdateTimer; // new timer for sync stats

  // Retry logic state
  int m_retryTargetCameraId; // camera currently retrying
  int m_retryMaxAttempts;
  int m_retryCurrentAttempt;
  QTimer *m_retryTimer; // schedules retry attempts non-blocking

  // Batch processing window
  stereo_vision::batch::BatchProcessingWindow* m_batchProcessingWindow;

  // Epipolar checker window
  EpipolarChecker* m_epipolarChecker;

  // AI Calibration state
  bool m_aiCalibrationActive;
  std::vector<cv::Mat> m_calibrationFramesLeft;
  std::vector<cv::Mat> m_calibrationFramesRight;
  int m_calibrationFrameCount;
  int m_requiredCalibrationFrames;
};

} // namespace stereo_vision::gui
