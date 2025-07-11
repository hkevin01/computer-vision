#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>

#include <memory>
#include <opencv2/opencv.hpp>

namespace stereo_vision {
class CameraManager;
class CameraCalibration;
} // namespace stereo_vision

namespace stereo_vision::gui {

class ImageDisplayWidget;

class CalibrationWizard : public QDialog {
  Q_OBJECT

public:
  explicit CalibrationWizard(
      std::shared_ptr<stereo_vision::CameraManager> cameraManager,
      QWidget *parent = nullptr);
  ~CalibrationWizard() override;

private slots:
  void nextStep();
  void previousStep();
  void captureFrame();
  void removeSelectedFrame();
  void clearAllFrames();
  void runCalibration();
  void onCameraFrameReceived();
  void onPatternTypeChanged();
  void onPatternSizeChanged();
  void finishWizard();
  void cancelWizard();

private:
  void setupUI();
  void setupWelcomePage();
  void setupPatternPage();
  void setupCapturePage();
  void setupReviewPage();
  void setupComputePage();
  void setupResultsPage();

  void updateNavigationButtons();
  void updateCaptureStatus();
  void updateQualityMetrics();
  bool detectCalibrationPattern(const cv::Mat &image,
                                std::vector<cv::Point2f> &corners);
  double calculateFrameQuality(const cv::Mat &image,
                               const std::vector<cv::Point2f> &corners);
  void addCalibratedFrame(const cv::Mat &image,
                          const std::vector<cv::Point2f> &corners);
  void updateFrameList();
  void showFrameDetails(int index);

  // UI Components
  QStackedWidget *m_stackedWidget;
  QPushButton *m_nextButton;
  QPushButton *m_previousButton;
  QPushButton *m_cancelButton;
  QPushButton *m_finishButton;
  QProgressBar *m_progressBar;
  QLabel *m_stepLabel;

  // Welcome page
  QWidget *m_welcomePage;
  QLabel *m_welcomeText;

  // Pattern configuration page
  QWidget *m_patternPage;
  QComboBox *m_patternTypeCombo;
  QSpinBox *m_patternWidthSpin;
  QSpinBox *m_patternHeightSpin;
  QDoubleSpinBox *m_squareSizeSpin;
  QLabel *m_patternPreview;

  // Capture page
  QWidget *m_capturePage;
  ImageDisplayWidget *m_liveView;
  QPushButton *m_captureButton;
  QLabel *m_detectionStatus;
  QLabel *m_qualityLabel;
  QLabel *m_frameCountLabel;
  QProgressBar *m_captureProgress;

  // Review page
  QWidget *m_reviewPage;
  QListWidget *m_frameList;
  ImageDisplayWidget *m_framePreview;
  QPushButton *m_removeFrameButton;
  QPushButton *m_clearAllButton;
  QLabel *m_frameInfoLabel;

  // Compute page
  QWidget *m_computePage;
  QProgressBar *m_computeProgress;
  QLabel *m_computeStatus;
  QTextEdit *m_computeLog;

  // Results page
  QWidget *m_resultsPage;
  QTextEdit *m_resultsText;
  QLabel *m_reprojectionError;
  QPushButton *m_saveButton;
  QPushButton *m_exportButton;

  // Data members
  std::shared_ptr<stereo_vision::CameraManager> m_cameraManager;
  std::shared_ptr<stereo_vision::CameraCalibration> m_calibration;

  int m_currentStep;
  static const int TOTAL_STEPS = 6;

  // Calibration data
  enum PatternType { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES };
  PatternType m_patternType;
  cv::Size m_patternSize;
  float m_squareSize;
  int m_requiredFrames;

  std::vector<cv::Mat> m_calibrationImages;
  std::vector<std::vector<cv::Point2f>> m_imagePoints;
  std::vector<std::vector<cv::Point3f>> m_objectPoints;
  std::vector<double> m_frameQualities;

  cv::Mat m_cameraMatrix;
  cv::Mat m_distortionCoeffs;
  std::vector<cv::Mat> m_rvecs;
  std::vector<cv::Mat> m_tvecs;
  double m_reprojectionErrorValue;

  // UI state
  bool m_isCapturing;
  bool m_patternDetected;
  QTimer *m_captureTimer;
  cv::Mat m_currentFrame;
  std::vector<cv::Point2f> m_currentCorners;
};

} // namespace stereo_vision::gui
