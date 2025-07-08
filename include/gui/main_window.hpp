#pragma once

#include <QAction>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include "camera_calibration.hpp"
#include "image_display_widget.hpp"
#include "parameter_panel.hpp"
#include "point_cloud_processor.hpp"
#include "point_cloud_widget.hpp"
#include "stereo_matcher.hpp"

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
  void onParameterChanged();
  void onProcessingFinished();

private:
  void setupUI();
  void setupMenuBar();
  void setupToolBar();
  void setupCentralWidget();
  void setupStatusBar();
  void connectSignals();
  void updateUI();
  void resetView();

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
  QAction *m_exportAction;
  QAction *m_aboutAction;

  // Status bar
  QStatusBar *m_statusBar;
  QProgressBar *m_progressBar;
  QLabel *m_statusLabel;

  // Processing components
  CameraCalibration *m_calibration;
  StereoMatcher *m_stereoMatcher;
  PointCloudProcessor *m_pointCloudProcessor;

  // Processing timer for async operations
  QTimer *m_processingTimer;

  // Current file paths
  QString m_leftImagePath;
  QString m_rightImagePath;
  QString m_calibrationPath;
  QString m_outputPath;

  // Processing state
  bool m_isProcessing;
  bool m_hasCalibration;
  bool m_hasImages;
};
