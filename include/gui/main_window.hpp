#pragma once

// System includes
#include <memory>

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
  void toggleProfiling(bool checked = false);

  // Webcam capture slots
  void showCameraSelector();
  void startWebcamCapture();
  void stopWebcamCapture();
  void captureLeftImage();
  void captureRightImage();
  void captureStereoImage();
  void onFrameReady();
  void onCameraSelectionChanged();

  // Live processing slots
  void toggleLiveProcessing();
  void onLiveFrameProcessed();
  void updateDisparityMap();
  void updatePointCloud();

  // AI Calibration slots
  void startAICalibration();
  void onCalibrationProgress(int progress);
  void onCalibrationComplete();
  void captureCalibrationFrame();

  // Batch processing slots
  void openBatchProcessing();

  // Epipolar checker slots
  void openEpipolarChecker();

private:
  void setupUI();
  void setupMenuBar();
  void setupCentralWidget();
  void setupStatusBar();
  void connectSignals();
  void updateUI();
  void resetView();
  void initializeCameraSystem();
  void refreshCameraStatus();
  void showCameraErrorDialog(const QString &title, const QString &message, const QString &details = "");
  void logCameraOperation(const QString &operation, bool success, const QString &details = "");
  void retryCameraConnection(int cameraId, int maxRetries = 3);
  void updateCameraStatusIndicators();

  // UI Components
  QWidget *m_centralWidget;
  QSplitter *m_mainSplitter;
  QTabWidget *m_imageTabWidget;

  // Image display
  ImageDisplayWidget *m_leftImageWidget;
  ImageDisplayWidget *m_rightImageWidget;
  ImageDisplayWidget *m_disparityWidget;

  // 3D visualization
  PointCloudWidget *m_pointCloudWidget;

  // Parameter controls
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
