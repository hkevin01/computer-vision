#include "gui/main_window.hpp"

// Standard library includes
#include <memory>

// Qt includes
#include <QAction>
#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QKeySequence>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// Project includes
#include "camera_calibration.hpp"
#include "camera_manager.hpp"
#include "gui/camera_selector_dialog.hpp"
#include "gui/image_display_widget.hpp"
#include "gui/parameter_panel.hpp"
#include "gui/point_cloud_widget.hpp"
#include "point_cloud_processor.hpp"
#include "stereo_matcher.hpp"

namespace stereo_vision::gui {

// Using declarations should be inside the namespace
using stereo_vision::CameraCalibration;
using stereo_vision::PointCloudProcessor;
using stereo_vision::StereoMatcher;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_centralWidget(nullptr), m_mainSplitter(nullptr),
      m_imageTabWidget(nullptr), m_leftImageWidget(nullptr),
      m_rightImageWidget(nullptr), m_disparityWidget(nullptr),
      m_pointCloudWidget(nullptr), m_parameterPanel(nullptr),
      m_progressBar(nullptr), m_statusLabel(nullptr), m_calibration(nullptr),
      m_stereoMatcher(nullptr), m_pointCloudProcessor(nullptr), m_cameraManager(nullptr),
      m_processingTimer(new QTimer(this)), m_captureTimer(new QTimer(this)),
      m_isProcessing(false), m_hasCalibration(false), m_hasImages(false),
      m_isCapturing(false), m_leftCameraConnected(false), m_rightCameraConnected(false),
      m_selectedLeftCamera(-1), m_selectedRightCamera(-1) {
  // Initialize processing components
  m_calibration = std::make_shared<CameraCalibration>();
  m_stereoMatcher = std::make_shared<StereoMatcher>();
  m_pointCloudProcessor = std::make_shared<PointCloudProcessor>();
  m_cameraManager = std::make_shared<stereo_vision::CameraManager>();

  setupUI();
  connectSignals();
  updateUI();

  // Set window properties
  setWindowTitle("Stereo Vision 3D Point Cloud Generator");
  setMinimumSize(1200, 800);
  resize(1600, 1000);

  // Set default output path
  m_outputPath =
      QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) +
      "/StereoVision";
  QDir().mkpath(m_outputPath);
}

MainWindow::~MainWindow() {
  // shared_ptr objects are automatically destroyed
}

void MainWindow::setupUI() {
  setupMenuBar();
  setupCentralWidget();
  setupStatusBar();
}

void MainWindow::setupMenuBar() {
  m_menuBar = menuBar();

  // File menu
  m_fileMenu = m_menuBar->addMenu("&File");

  m_openLeftAction = new QAction("Open &Left Image...", this);
  m_openLeftAction->setShortcut(QKeySequence("Ctrl+L"));
  m_fileMenu->addAction(m_openLeftAction);

  m_openRightAction = new QAction("Open &Right Image...", this);
  m_openRightAction->setShortcut(QKeySequence("Ctrl+R"));
  m_fileMenu->addAction(m_openRightAction);

  m_openFolderAction = new QAction("Open Stereo &Folder...", this);
  m_openFolderAction->setShortcut(QKeySequence("Ctrl+F"));
  m_fileMenu->addAction(m_openFolderAction);

  m_fileMenu->addSeparator();
  
  // Webcam capture menu
  m_cameraSelectAction = new QAction("Select &Cameras...", this);
  m_cameraSelectAction->setShortcut(QKeySequence("Ctrl+Shift+C"));
  m_fileMenu->addAction(m_cameraSelectAction);
  
  m_startCaptureAction = new QAction("&Start Webcam Capture", this);
  m_startCaptureAction->setShortcut(QKeySequence("Ctrl+Shift+S"));
  m_startCaptureAction->setEnabled(false);
  m_fileMenu->addAction(m_startCaptureAction);
  
  m_stopCaptureAction = new QAction("S&top Webcam Capture", this);
  m_stopCaptureAction->setShortcut(QKeySequence("Ctrl+Shift+T"));
  m_stopCaptureAction->setEnabled(false);
  m_fileMenu->addAction(m_stopCaptureAction);
  
  m_captureLeftAction = new QAction("Capture &Left Image", this);
  m_captureLeftAction->setShortcut(QKeySequence("L"));
  m_captureLeftAction->setEnabled(false);
  m_fileMenu->addAction(m_captureLeftAction);
  
  m_captureRightAction = new QAction("Capture &Right Image", this);
  m_captureRightAction->setShortcut(QKeySequence("R"));
  m_captureRightAction->setEnabled(false);
  m_fileMenu->addAction(m_captureRightAction);
  
  m_captureStereoAction = new QAction("Capture &Stereo Pair", this);
  m_captureStereoAction->setShortcut(QKeySequence("Space"));
  m_captureStereoAction->setEnabled(false);
  m_fileMenu->addAction(m_captureStereoAction);
  
  m_fileMenu->addSeparator();

  m_loadCalibrationAction = new QAction("&Load Calibration...", this);
  m_loadCalibrationAction->setShortcut(QKeySequence("Ctrl+O"));
  m_fileMenu->addAction(m_loadCalibrationAction);

  m_saveCalibrationAction = new QAction("&Save Calibration...", this);
  m_saveCalibrationAction->setShortcut(QKeySequence("Ctrl+S"));
  m_fileMenu->addAction(m_saveCalibrationAction);

  m_fileMenu->addSeparator();

  m_exitAction = new QAction("E&xit", this);
  m_exitAction->setShortcut(QKeySequence("Ctrl+Q"));
  m_fileMenu->addAction(m_exitAction);

  // Process menu
  m_processMenu = m_menuBar->addMenu("&Process");

  m_calibrateAction = new QAction("&Calibrate Cameras...", this);
  m_calibrateAction->setShortcut(QKeySequence("Ctrl+C"));
  m_processMenu->addAction(m_calibrateAction);

  m_processAction = new QAction("Process &Stereo Images", this);
  m_processAction->setShortcut(QKeySequence("Ctrl+P"));
  m_processMenu->addAction(m_processAction);

  m_exportAction = new QAction("&Export Point Cloud...", this);
  m_exportAction->setShortcut(QKeySequence("Ctrl+E"));
  m_processMenu->addAction(m_exportAction);

  // View menu
  m_viewMenu = m_menuBar->addMenu("&View");

  // Help menu
  m_helpMenu = m_menuBar->addMenu("&Help");

  m_aboutAction = new QAction("&About", this);
  m_helpMenu->addAction(m_aboutAction);
}

void MainWindow::setupCentralWidget() {
  m_centralWidget = new QWidget;
  setCentralWidget(m_centralWidget);

  // Main splitter
  m_mainSplitter = new QSplitter(Qt::Horizontal);

  // Left side: Image display and 3D view
  auto *leftWidget = new QWidget;
  auto *leftLayout = new QVBoxLayout(leftWidget);

  // Image tab widget
  m_imageTabWidget = new QTabWidget;

  m_leftImageWidget = new ImageDisplayWidget;
  m_rightImageWidget = new ImageDisplayWidget;
  m_disparityWidget = new ImageDisplayWidget;

  m_imageTabWidget->addTab(m_leftImageWidget, "Left Image");
  m_imageTabWidget->addTab(m_rightImageWidget, "Right Image");
  m_imageTabWidget->addTab(m_disparityWidget, "Disparity Map");

  // Point cloud widget
  m_pointCloudWidget = new PointCloudWidget;

  // Add to left layout
  leftLayout->addWidget(m_imageTabWidget, 3);
  leftLayout->addWidget(m_pointCloudWidget, 2);

  // Right side: Parameter panel
  m_parameterPanel = new ParameterPanel;

  // Add to main splitter
  m_mainSplitter->addWidget(leftWidget);
  m_mainSplitter->addWidget(m_parameterPanel);
  m_mainSplitter->setStretchFactor(0, 3);
  m_mainSplitter->setStretchFactor(1, 1);

  // Main layout
  auto *mainLayout = new QVBoxLayout(m_centralWidget);
  mainLayout->addWidget(m_mainSplitter);
}

void MainWindow::setupStatusBar() {
  m_statusBar = statusBar();

  m_statusLabel = new QLabel("Ready");
  m_statusBar->addWidget(m_statusLabel);

  m_progressBar = new QProgressBar;
  m_progressBar->setVisible(false);
  m_statusBar->addPermanentWidget(m_progressBar);
}

void MainWindow::connectSignals() {
  // File actions
  connect(m_openLeftAction, &QAction::triggered, this,
          &MainWindow::openLeftImage);
  connect(m_openRightAction, &QAction::triggered, this,
          &MainWindow::openRightImage);
  connect(m_openFolderAction, &QAction::triggered, this,
          &MainWindow::openStereoFolder);
  connect(m_loadCalibrationAction, &QAction::triggered, this,
          &MainWindow::loadCalibration);
  connect(m_saveCalibrationAction, &QAction::triggered, this,
          &MainWindow::saveCalibration);
  connect(m_exitAction, &QAction::triggered, this, &QWidget::close);

  // Process actions
  connect(m_calibrateAction, &QAction::triggered, this,
          &MainWindow::runCalibration);
  connect(m_processAction, &QAction::triggered, this,
          &MainWindow::processStereoImages);
  connect(m_exportAction, &QAction::triggered, this,
          &MainWindow::exportPointCloud);

  // Help actions
  connect(m_aboutAction, &QAction::triggered, this, &MainWindow::showAbout);

  // Webcam capture actions
  connect(m_cameraSelectAction, &QAction::triggered, this, 
          &MainWindow::showCameraSelector);
  connect(m_startCaptureAction, &QAction::triggered, this, 
          &MainWindow::startWebcamCapture);
  connect(m_stopCaptureAction, &QAction::triggered, this, 
          &MainWindow::stopWebcamCapture);
  connect(m_captureLeftAction, &QAction::triggered, this, 
          &MainWindow::captureLeftImage);
  connect(m_captureRightAction, &QAction::triggered, this, 
          &MainWindow::captureRightImage);
  connect(m_captureStereoAction, &QAction::triggered, this, 
          &MainWindow::captureStereoImage);

  // Parameter panel
  connect(m_parameterPanel, &ParameterPanel::parametersChanged, this,
          &MainWindow::onParameterChanged);

  // Processing timer
  connect(m_processingTimer, &QTimer::timeout, this,
          &MainWindow::onProcessingFinished);
}

void MainWindow::openLeftImage() {
  QString fileName = QFileDialog::getOpenFileName(
      this, "Open Left Image", m_leftImagePath,
      "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)");

  if (!fileName.isEmpty()) {
    m_leftImagePath = fileName;
    m_leftImageWidget->setImage(fileName);
    m_hasImages = !m_rightImagePath.isEmpty();
    updateUI();
    m_statusLabel->setText("Left image loaded: " +
                           QFileInfo(fileName).fileName());
  }
}

void MainWindow::openRightImage() {
  QString fileName = QFileDialog::getOpenFileName(
      this, "Open Right Image", m_rightImagePath,
      "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)");

  if (!fileName.isEmpty()) {
    m_rightImagePath = fileName;
    m_rightImageWidget->setImage(fileName);
    m_hasImages = !m_leftImagePath.isEmpty();
    updateUI();
    m_statusLabel->setText("Right image loaded: " +
                           QFileInfo(fileName).fileName());
  }
}

void MainWindow::openStereoFolder() {
  QString dirName = QFileDialog::getExistingDirectory(
      this, "Open Stereo Image Folder", m_outputPath);

  if (!dirName.isEmpty()) {
    QDir dir(dirName);
    QStringList imageFiles = dir.entryList(
        QStringList() << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.tiff",
        QDir::Files, QDir::Name);

    if (imageFiles.size() >= 2) {
      m_leftImagePath = dir.absoluteFilePath(imageFiles[0]);
      m_rightImagePath = dir.absoluteFilePath(imageFiles[1]);

      m_leftImageWidget->setImage(m_leftImagePath);
      m_rightImageWidget->setImage(m_rightImagePath);

      m_hasImages = true;
      updateUI();
      m_statusLabel->setText(
          QString("Loaded stereo pair from: %1").arg(dir.dirName()));
    } else {
      QMessageBox::warning(this, "Error",
                           "Folder must contain at least 2 image files.");
    }
  }
}

void MainWindow::loadCalibration() {
  QString fileName = QFileDialog::getOpenFileName(
      this, tr("Load Calibration File"), "",
      tr("XML/YAML files (*.xml *.yaml *.yml);;All files (*.*)"));
  if (fileName.isEmpty()) {
    return;
  }

  if (m_calibration->loadCalibration(fileName.toStdString(), m_stereoParams)) {
    m_statusLabel->setText(tr("Calibration loaded successfully."));
    updateUI();
  } else {
    m_hasCalibration = false;
    QMessageBox::critical(this, tr("Error"),
                          tr("Failed to load calibration file."));
  }
}

void MainWindow::saveCalibration() {
  QString fileName = QFileDialog::getSaveFileName(
      this, tr("Save Calibration File"), "",
      tr("XML/YAML files (*.xml *.yaml *.yml);;All files (*.*)"));
  if (fileName.isEmpty()) {
    return;
  }

  if (m_calibration->saveCalibration(fileName.toStdString(), m_stereoParams)) {
    m_statusLabel->setText(tr("Calibration saved successfully."));
  } else {
    QMessageBox::critical(this, tr("Error"),
                          tr("Failed to save calibration file."));
  }
}

void MainWindow::runCalibration() {
  // TODO: Implement calibration dialog
  QMessageBox::information(this, "Calibration",
                           "Camera calibration feature coming soon!");
}

void MainWindow::processStereoImages() {
  if (!m_hasImages || !m_hasCalibration) {
    QMessageBox::warning(
        this, "Error", "Please load stereo images and calibration data first.");
    return;
  }

  m_isProcessing = true;
  updateUI();

  m_statusLabel->setText("Processing stereo images...");
  m_progressBar->setVisible(true);
  m_progressBar->setRange(0, 0); // Indeterminate progress

  // Start processing timer (simulating async processing)
  m_processingTimer->start(2000); // 2 seconds delay for demo
}

void MainWindow::exportPointCloud() {
  QString fileName = QFileDialog::getSaveFileName(
      this, "Export Point Cloud", m_outputPath + "/pointcloud.ply",
      "Point Cloud Files (*.ply *.pcd *.xyz)");

  if (!fileName.isEmpty()) {
    // TODO: Implement actual export
    m_statusLabel->setText("Point cloud exported: " +
                           QFileInfo(fileName).fileName());
  }
}

void MainWindow::showAbout() {
  QMessageBox::about(this, "About Stereo Vision",
                     "<h3>Stereo Vision 3D Point Cloud Generator</h3>"
                     "<p>A high-performance C++ application for generating 3D "
                     "point clouds from stereo camera images.</p>"
                     "<p><b>Features:</b></p>"
                     "<ul>"
                     "<li>GPU-accelerated stereo matching (CUDA/HIP)</li>"
                     "<li>Camera calibration</li>"
                     "<li>Real-time processing</li>"
                     "<li>Multiple export formats</li>"
                     "</ul>"
                     "<p><b>Version:</b> 1.0.0</p>"
                     "<p><b>Build:</b> " +
                         QString(__DATE__) + " " + QString(__TIME__) + "</p>");
}

void MainWindow::onParameterChanged() {
  // TODO: Reprocess if images are loaded
  m_statusLabel->setText("Parameters updated");
}

void MainWindow::onProcessingFinished() {
  m_processingTimer->stop();
  m_isProcessing = false;

  m_progressBar->setVisible(false);
  m_statusLabel->setText("Processing completed");

  // TODO: Display results
  updateUI();
}

void MainWindow::updateUI() {
  // Update action states
  m_processAction->setEnabled(m_hasImages && m_hasCalibration &&
                              !m_isProcessing);
  m_exportAction->setEnabled(!m_isProcessing);
  m_saveCalibrationAction->setEnabled(m_hasCalibration && !m_isProcessing);

  // Update window title
  QString title = "Stereo Vision 3D Point Cloud Generator";
  if (m_hasImages) {
    title += " - Images Loaded";
  }
  if (m_hasCalibration) {
    title += " - Calibrated";
  }
  if (m_isProcessing) {
    title += " - Processing...";
  }
  setWindowTitle(title);
}

void MainWindow::resetView() {
  m_leftImageWidget->clearImage();
  m_rightImageWidget->clearImage();
  m_disparityWidget->clearImage();
  m_pointCloudWidget->clearPointCloud();

  m_leftImagePath.clear();
  m_rightImagePath.clear();
  m_hasImages = false;

  updateUI();
  m_statusLabel->setText("View reset");
}

// Webcam capture implementation
void MainWindow::showCameraSelector() {
  auto dialog = new CameraSelectorDialog(m_cameraManager, this);
  
  if (dialog->exec() == QDialog::Accepted && dialog->areCamerasConfigured()) {
    m_selectedLeftCamera = dialog->getSelectedLeftCamera();
    m_selectedRightCamera = dialog->getSelectedRightCamera();
    
    // Update camera connection status
    m_leftCameraConnected = (m_selectedLeftCamera >= 0);
    m_rightCameraConnected = (m_selectedRightCamera >= 0);
    
    // Enable capture controls if cameras are selected
    m_startCaptureAction->setEnabled(m_leftCameraConnected || m_rightCameraConnected);
    
    m_statusLabel->setText(QString("Cameras configured: Left=%1, Right=%2")
                          .arg(m_selectedLeftCamera).arg(m_selectedRightCamera));
  } else {
    m_statusLabel->setText("Camera selection cancelled");
  }
  
  dialog->deleteLater();
}

void MainWindow::startWebcamCapture() {
  if (!m_leftCameraConnected && !m_rightCameraConnected) {
    QMessageBox::warning(this, "No Cameras", 
                        "Please select cameras first using 'Select Cameras...'");
    return;
  }
  
  // Open cameras using the camera manager
  if (!m_cameraManager->openCameras(m_selectedLeftCamera, m_selectedRightCamera)) {
    QMessageBox::critical(this, "Camera Error", 
                         "Failed to open selected cameras. Please check connections.");
    return;
  }
  
  // Start capture timer for live preview
  m_captureTimer->start(33); // ~30 FPS
  m_isCapturing = true;
  
  // Update UI
  m_startCaptureAction->setEnabled(false);
  m_stopCaptureAction->setEnabled(true);
  m_captureLeftAction->setEnabled(m_leftCameraConnected);
  m_captureRightAction->setEnabled(m_rightCameraConnected);
  m_captureStereoAction->setEnabled(m_leftCameraConnected && m_rightCameraConnected);
  
  // Connect capture timer
  connect(m_captureTimer, &QTimer::timeout, this, &MainWindow::onFrameReady);
  
  m_statusLabel->setText("Webcam capture started - live preview active");
}

void MainWindow::stopWebcamCapture() {
  if (!m_isCapturing) return;
  
  // Stop timer and disconnect
  m_captureTimer->stop();
  disconnect(m_captureTimer, &QTimer::timeout, this, &MainWindow::onFrameReady);
  
  // Close cameras
  m_cameraManager->closeCameras();
  m_isCapturing = false;
  
  // Update UI
  m_startCaptureAction->setEnabled(m_leftCameraConnected || m_rightCameraConnected);
  m_stopCaptureAction->setEnabled(false);
  m_captureLeftAction->setEnabled(false);
  m_captureRightAction->setEnabled(false);
  m_captureStereoAction->setEnabled(false);
  
  m_statusLabel->setText("Webcam capture stopped");
}

void MainWindow::captureLeftImage() {
  if (!m_isCapturing || m_lastLeftFrame.empty()) {
    QMessageBox::warning(this, "Capture Error", 
                        "No left camera frame available. Ensure webcam capture is running.");
    return;
  }
  
  // Save the current left frame
  QString fileName = QFileDialog::getSaveFileName(
      this, "Save Left Image", 
      m_outputPath + "/left_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
      "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)");
  
  if (!fileName.isEmpty()) {
    if (cv::imwrite(fileName.toStdString(), m_lastLeftFrame)) {
      // Also load it as the current left image
      m_leftImagePath = fileName;
      m_leftImageWidget->setImage(fileName);
      m_hasImages = !m_rightImagePath.isEmpty();
      updateUI();
      
      m_statusLabel->setText("Left image captured: " + QFileInfo(fileName).fileName());
    } else {
      QMessageBox::critical(this, "Save Error", "Failed to save left image.");
    }
  }
}

void MainWindow::captureRightImage() {
  if (!m_isCapturing || m_lastRightFrame.empty()) {
    QMessageBox::warning(this, "Capture Error", 
                        "No right camera frame available. Ensure webcam capture is running.");
    return;
  }
  
  // Save the current right frame
  QString fileName = QFileDialog::getSaveFileName(
      this, "Save Right Image", 
      m_outputPath + "/right_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
      "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)");
  
  if (!fileName.isEmpty()) {
    if (cv::imwrite(fileName.toStdString(), m_lastRightFrame)) {
      // Also load it as the current right image
      m_rightImagePath = fileName;
      m_rightImageWidget->setImage(fileName);
      m_hasImages = !m_leftImagePath.isEmpty();
      updateUI();
      
      m_statusLabel->setText("Right image captured: " + QFileInfo(fileName).fileName());
    } else {
      QMessageBox::critical(this, "Save Error", "Failed to save right image.");
    }
  }
}

void MainWindow::captureStereoImage() {
  if (!m_isCapturing || m_lastLeftFrame.empty() || m_lastRightFrame.empty()) {
    QMessageBox::warning(this, "Capture Error", 
                        "Both camera frames must be available. Ensure webcam capture is running.");
    return;
  }
  
  // Create timestamp for synchronized capture
  QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
  
  // Save both frames simultaneously
  QString leftFileName = m_outputPath + "/left_" + timestamp + ".png";
  QString rightFileName = m_outputPath + "/right_" + timestamp + ".png";
  
  QDir().mkpath(m_outputPath); // Ensure directory exists
  
  bool leftSaved = cv::imwrite(leftFileName.toStdString(), m_lastLeftFrame);
  bool rightSaved = cv::imwrite(rightFileName.toStdString(), m_lastRightFrame);
  
  if (leftSaved && rightSaved) {
    // Load both images into the interface
    m_leftImagePath = leftFileName;
    m_rightImagePath = rightFileName;
    m_leftImageWidget->setImage(leftFileName);
    m_rightImageWidget->setImage(rightFileName);
    m_hasImages = true;
    updateUI();
    
    m_statusLabel->setText("Stereo pair captured: " + timestamp);
  } else {
    QMessageBox::critical(this, "Save Error", 
                         "Failed to save stereo image pair. Check output directory permissions.");
  }
}

void MainWindow::onFrameReady() {
  if (!m_isCapturing || !m_cameraManager->areCamerasOpen()) {
    return;
  }
  
  cv::Mat leftFrame, rightFrame;
  
  // Grab frames from cameras
  if (m_cameraManager->grabFrames(leftFrame, rightFrame)) {
    // Store the latest frames
    if (!leftFrame.empty()) {
      m_lastLeftFrame = leftFrame.clone();
    }
    if (!rightFrame.empty()) {
      m_lastRightFrame = rightFrame.clone();
    }
    
    // Update live preview in the image widgets (optional - shows live feed)
    if (!leftFrame.empty() && m_leftCameraConnected) {
      // Convert OpenCV Mat to QImage and display
      cv::Mat displayFrame;
      cv::cvtColor(leftFrame, displayFrame, cv::COLOR_BGR2RGB);
      
      QImage qimg(displayFrame.data, displayFrame.cols, displayFrame.rows, 
                  displayFrame.step, QImage::Format_RGB888);
      
      // Save as temporary file and load (this could be optimized)
      QString tempPath = QDir::tempPath() + "/stereo_left_preview.png";
      qimg.save(tempPath);
      m_leftImageWidget->setImage(tempPath);
    }
    
    if (!rightFrame.empty() && m_rightCameraConnected) {
      // Convert OpenCV Mat to QImage and display
      cv::Mat displayFrame;
      cv::cvtColor(rightFrame, displayFrame, cv::COLOR_BGR2RGB);
      
      QImage qimg(displayFrame.data, displayFrame.cols, displayFrame.rows, 
                  displayFrame.step, QImage::Format_RGB888);
      
      // Save as temporary file and load (this could be optimized)
      QString tempPath = QDir::tempPath() + "/stereo_right_preview.png";
      qimg.save(tempPath);
      m_rightImageWidget->setImage(tempPath);
    }
  }
}

void MainWindow::onCameraSelectionChanged() {
  // This slot can be used for future dynamic camera switching
  updateUI();
}

} // namespace stereo_vision::gui
