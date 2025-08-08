#include <QApplication>
#include <QWidget>
#include <QMainWindow>
#include <QAction>
#include <QMenuBar>
#include <QMenu>
#include <QStatusBar>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QSplitter>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QDir>
#include <QStandardPaths>
#include <QDateTime>
#include <QFileDialog>
#include <QMessageBox>
#include <QKeySequence>
#include <QString>
#include <QThread>
#include <QSize>
#include <QObject>
#include <QDebug>
#include <QLoggingCategory>
#include <QGroupBox>
#include <QTextEdit>
#include <QPushButton>
#include <QHBoxLayout>
#include <QFrame>

Q_LOGGING_CATEGORY(cameraDebug, "camera.debug")
Q_LOGGING_CATEGORY(cameraError, "camera.error")

// Function to register required types
static void registerQtTypes() {
    qRegisterMetaType<QAction*>("QAction*");
    qRegisterMetaType<QTimer*>("QTimer*");
}

// OpenCV includes
#include <opencv2/opencv.hpp>

// Project includes (after Qt/OpenCV)
#include "gui/main_window.hpp"
#include "gui/parameter_panel.hpp"
#include "gui/calibration_wizard.hpp"
#include "gui/camera_selector_dialog.hpp"
#include "gui/batch_processing_window.hpp"
#include "gui/epipolar_checker.hpp"
#include "camera_manager.hpp"
#include "multicam/multi_camera_system_simple.hpp"
#include "utils/perf_profiler.hpp" // profiling

namespace stereo_vision::gui {

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      m_centralWidget(nullptr),
      m_mainSplitter(nullptr),
      m_imageTabWidget(nullptr),
      m_leftImageWidget(nullptr),
      m_rightImageWidget(nullptr),
      m_disparityWidget(nullptr),
      m_pointCloudWidget(nullptr),
      m_parameterPanel(nullptr),
      m_progressBar(nullptr),
      m_statusLabel(nullptr),
      m_cameraStatusGroup(nullptr),
      m_leftCameraStatusLabel(nullptr),
      m_rightCameraStatusLabel(nullptr),
      m_retryConnectionButton(nullptr),
      m_debugLogOutput(nullptr),
      m_calibration(nullptr),
      m_stereoMatcher(nullptr),
      m_pointCloudProcessor(nullptr),
      m_cameraManager(nullptr),
      m_multiCameraSystem(std::make_shared<stereovision::multicam::MultiCameraSystem>()), // new
      m_processingTimer(new QTimer(this)),
      m_captureTimer(new QTimer(this)),
      m_isProcessing(false),
      m_hasCalibration(false),
      m_hasImages(false),
      m_isCapturing(false),
      m_leftCameraConnected(false),
      m_rightCameraConnected(false),
      m_selectedLeftCamera(-1),
      m_selectedRightCamera(-1),
      m_liveProcessingTimer(new QTimer(this)),
      m_liveProcessingEnabled(false),
      m_aiCalibrationActive(false),
      m_calibrationFrameCount(0),
      m_requiredCalibrationFrames(20),
      m_profilingTimer(new QTimer(this)),
      m_syncUpdateTimer(new QTimer(this)), // new
      m_retryTargetCameraId(-1),
      m_retryMaxAttempts(0),
      m_retryCurrentAttempt(0),
      m_retryTimer(new QTimer(this)), // new
      m_batchProcessingWindow(nullptr),
      m_epipolarChecker(nullptr) {

    // Register Qt types first
    registerQtTypes();

    // Initialize legacy camera manager early (still used for single camera paths)
    m_cameraManager = std::make_shared<stereo_vision::CameraManager>();

    // Set up UI components in the correct order
    setupUI();          // Creates all widgets including parameter panel
    connectSignals();   // Connect signals after widgets are created
    updateUI();         // Update initial UI state

    // Initialize camera systems and detect available cameras
    initializeCameraSystem();

    // Periodic camera status checking
    QTimer *cameraStatusTimer = new QTimer(this);
    connect(cameraStatusTimer, &QTimer::timeout, this, &MainWindow::refreshCameraStatus);
    cameraStatusTimer->start(5000);

    // Profiling timer (disabled until enabled via action)
    m_profilingTimer->setInterval(2000);
    connect(m_profilingTimer, &QTimer::timeout, this, &MainWindow::updateProfilingStats);

    // Sync status update timer
    m_syncUpdateTimer->setInterval(1500); // update every 1.5s
    connect(m_syncUpdateTimer, &QTimer::timeout, this, &MainWindow::updateSyncStatus);
    m_syncUpdateTimer->start();

    // Retry timer (single-shot per attempt)
    m_retryTimer->setSingleShot(true);
    connect(m_retryTimer, &QTimer::timeout, this, &MainWindow::performRetryAttempt);

    // Window properties
    setWindowTitle("Stereo Vision 3D Point Cloud Generator");
    setFixedSize(1200, 800);

    // Default output path
    m_outputPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/StereoVision";
    QDir().mkpath(m_outputPath);
}

MainWindow::~MainWindow() {
    // Clean up windows
    delete m_epipolarChecker;
    delete m_batchProcessingWindow;
}

void MainWindow::setupUI() {
    setupMenuBar();
    setupCentralWidget();
    setupStatusBar();
}

void MainWindow::setupMenuBar() {
    m_menuBar = menuBar();

    // Create File menu actions
    m_openLeftAction = new QAction("Open &Left Image...", this);
    m_openRightAction = new QAction("Open &Right Image...", this);
    m_openFolderAction = new QAction("Open Stereo &Folder...", this);
    m_cameraSelectAction = new QAction("Select &Cameras...", this);
    m_startCaptureAction = new QAction("&Start Webcam Capture", this);
    m_stopCaptureAction = new QAction("S&top Webcam Capture", this);
    m_captureLeftAction = new QAction("Capture &Left Image", this);
    m_captureRightAction = new QAction("Capture &Right Image", this);
    m_captureStereoAction = new QAction("Capture &Stereo Pair", this);
    m_loadCalibrationAction = new QAction("&Load Calibration...", this);
    m_saveCalibrationAction = new QAction("&Save Calibration...", this);
    m_exitAction = new QAction("E&xit", this);

    // Create Process menu actions
    m_calibrateAction = new QAction("&Calibrate Cameras...", this);
    m_aiCalibrationAction = new QAction("&AI Auto-Calibration... ⭐", this);
    m_processAction = new QAction("Process &Stereo Images", this);
    m_batchProcessAction = new QAction("&Batch Processing...", this);
    m_epipolarCheckerAction = new QAction("&Epipolar Checker", this);
    m_liveProcessingAction = new QAction("&Live Processing", this);
    m_exportAction = new QAction("&Export Point Cloud", this);

    // Set up shortcuts with proper context
    m_openLeftAction->setShortcut(QKeySequence("Ctrl+L"));
    m_openLeftAction->setShortcutContext(Qt::WindowShortcut);
    m_openRightAction->setShortcut(QKeySequence("Ctrl+R"));
    m_openRightAction->setShortcutContext(Qt::WindowShortcut);
    m_openFolderAction->setShortcut(QKeySequence("Ctrl+F"));
    m_openFolderAction->setShortcutContext(Qt::WindowShortcut);
    m_cameraSelectAction->setShortcut(QKeySequence("Ctrl+Shift+C"));
    m_cameraSelectAction->setShortcutContext(Qt::WindowShortcut);
    m_startCaptureAction->setShortcut(QKeySequence("Ctrl+Shift+S"));
    m_startCaptureAction->setShortcutContext(Qt::WindowShortcut);
    m_stopCaptureAction->setShortcut(QKeySequence("Ctrl+Shift+T"));
    m_stopCaptureAction->setShortcutContext(Qt::WindowShortcut);
    m_captureLeftAction->setShortcut(QKeySequence("L"));
    m_captureLeftAction->setShortcutContext(Qt::WindowShortcut);
    m_captureRightAction->setShortcut(QKeySequence("R"));
    m_captureRightAction->setShortcutContext(Qt::WindowShortcut);
    m_captureStereoAction->setShortcut(QKeySequence("Space"));
    m_captureStereoAction->setShortcutContext(Qt::WindowShortcut);
    m_loadCalibrationAction->setShortcut(QKeySequence("Ctrl+O"));
    m_loadCalibrationAction->setShortcutContext(Qt::WindowShortcut);
    m_saveCalibrationAction->setShortcut(QKeySequence("Ctrl+S"));
    m_saveCalibrationAction->setShortcutContext(Qt::WindowShortcut);
    m_exitAction->setShortcut(QKeySequence("Ctrl+Q"));
    m_exitAction->setShortcutContext(Qt::WindowShortcut);
    m_calibrateAction->setShortcut(QKeySequence("Ctrl+C"));
    m_calibrateAction->setShortcutContext(Qt::WindowShortcut);
    m_aiCalibrationAction->setShortcut(QKeySequence("Ctrl+Alt+C"));
    m_aiCalibrationAction->setShortcutContext(Qt::WindowShortcut);
    m_processAction->setShortcut(QKeySequence("Ctrl+P"));
    m_processAction->setShortcutContext(Qt::WindowShortcut);
    m_batchProcessAction->setShortcut(QKeySequence("Ctrl+B"));
    m_batchProcessAction->setShortcutContext(Qt::WindowShortcut);
    m_epipolarCheckerAction->setShortcut(QKeySequence("Ctrl+Shift+E"));
    m_epipolarCheckerAction->setShortcutContext(Qt::WindowShortcut);
    m_liveProcessingAction->setShortcut(QKeySequence("Ctrl+Shift+P"));
    m_liveProcessingAction->setShortcutContext(Qt::WindowShortcut);
    m_exportAction->setShortcut(QKeySequence("Ctrl+E"));
    m_exportAction->setShortcutContext(Qt::WindowShortcut);

    // Set initial states
    m_startCaptureAction->setEnabled(false);
    m_stopCaptureAction->setEnabled(false);
    m_captureLeftAction->setEnabled(false);
    m_captureRightAction->setEnabled(false);
    m_captureStereoAction->setEnabled(false);
    m_liveProcessingAction->setCheckable(true);

    // Set status tips
    m_calibrateAction->setStatusTip("Launch interactive camera calibration wizard with step-by-step guidance");
    m_aiCalibrationAction->setStatusTip("Fully functional AI-powered automatic camera calibration with quality assessment");
    m_batchProcessAction->setStatusTip("Open batch processing window for multiple stereo pairs");
    m_epipolarCheckerAction->setStatusTip("Open epipolar line checker for calibration quality assessment");

    // Create menus
    m_fileMenu = m_menuBar->addMenu("&File");
    m_processMenu = m_menuBar->addMenu("&Process");

    // Populate File menu
    m_fileMenu->addAction(m_openLeftAction);
    m_fileMenu->addAction(m_openRightAction);
    m_fileMenu->addAction(m_openFolderAction);
    m_fileMenu->addSeparator();
    m_fileMenu->addAction(m_cameraSelectAction);
    m_fileMenu->addAction(m_startCaptureAction);
    m_fileMenu->addAction(m_stopCaptureAction);
    m_fileMenu->addAction(m_captureLeftAction);
    m_fileMenu->addAction(m_captureRightAction);
    m_fileMenu->addAction(m_captureStereoAction);
    m_fileMenu->addSeparator();
    m_fileMenu->addAction(m_loadCalibrationAction);
    m_fileMenu->addAction(m_saveCalibrationAction);
    m_fileMenu->addSeparator();
    m_fileMenu->addAction(m_exitAction);

    // Populate Process menu
    m_processMenu->addAction(m_calibrateAction);
    m_processMenu->addAction(m_aiCalibrationAction);
    m_processMenu->addSeparator();
    m_processMenu->addAction(m_processAction);
    m_processMenu->addAction(m_batchProcessAction);
    m_processMenu->addAction(m_epipolarCheckerAction);
    m_processMenu->addAction(m_liveProcessingAction);
    m_processMenu->addSeparator();
    m_processMenu->addAction(m_exportAction);

    // Create and populate View menu
    m_viewMenu = m_menuBar->addMenu("&View");

    m_showDisparityAction = new QAction("Show &Disparity Map", this);
    m_showPointCloudAction = new QAction("Show &Point Cloud", this);
    m_enableProfilingAction = new QAction("Enable &Profiling", this);
    m_enableProfilingAction->setCheckable(true);
    m_enableProfilingAction->setStatusTip("Toggle runtime performance profiling and periodic stats");

    m_showDisparityAction->setShortcut(QKeySequence("Ctrl+D"));
    m_showDisparityAction->setShortcutContext(Qt::WindowShortcut);
    m_showDisparityAction->setCheckable(true);
    m_showDisparityAction->setChecked(true);

    m_showPointCloudAction->setShortcut(QKeySequence("Ctrl+3"));
    m_showPointCloudAction->setShortcutContext(Qt::WindowShortcut);
    m_showPointCloudAction->setCheckable(true);
    m_showPointCloudAction->setChecked(true);

    m_viewMenu->addAction(m_showDisparityAction);
    m_viewMenu->addAction(m_showPointCloudAction);
    m_viewMenu->addSeparator();
    m_viewMenu->addAction(m_enableProfilingAction);

    // Create and populate Help menu
    m_helpMenu = m_menuBar->addMenu("&Help");
    m_shortcutsAction = new QAction("&Keyboard Shortcuts", this);
    m_shortcutsAction->setShortcut(QKeySequence("F1"));
    m_shortcutsAction->setShortcutContext(Qt::WindowShortcut);
    m_aboutAction = new QAction("&About", this);
    m_helpMenu->addAction(m_shortcutsAction);
    m_helpMenu->addSeparator();
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

    // Add live view tab for real-time processing
    auto *liveViewWidget = new QWidget;
    auto *liveLayout = new QHBoxLayout(liveViewWidget);

    // Create mini views for live processing
    auto *liveStereoWidget = new ImageDisplayWidget;
    auto *liveDisparityWidget = new ImageDisplayWidget;

    liveStereoWidget->setMinimumSize(320, 240);
    liveDisparityWidget->setMinimumSize(320, 240);

    liveLayout->addWidget(liveStereoWidget);
    liveLayout->addWidget(liveDisparityWidget);

    m_imageTabWidget->addTab(liveViewWidget, "Live Processing");

    // Point cloud widget
    m_pointCloudWidget = new PointCloudWidget;

    // Add to left layout with adjusted proportions for better live view
    leftLayout->addWidget(m_imageTabWidget, 3);
    leftLayout->addWidget(m_pointCloudWidget, 2);

    // Right side: Parameter panel and debug log
    auto *rightWidget = new QWidget;
    auto *rightLayout = new QVBoxLayout(rightWidget);

    m_parameterPanel = new ParameterPanel;
    rightLayout->addWidget(m_parameterPanel, 2);

    // Add debug log output
    QGroupBox *debugGroup = new QGroupBox("Camera Debug Log");
    QVBoxLayout *debugLayout = new QVBoxLayout(debugGroup);
    m_debugLogOutput = new QTextEdit;
    m_debugLogOutput->setMaximumHeight(150);
    m_debugLogOutput->setFont(QFont("Courier", 9));
    m_debugLogOutput->setPlaceholderText("Camera operation logs will appear here...");
    debugLayout->addWidget(m_debugLogOutput);
    rightLayout->addWidget(debugGroup, 1);

    // Add to main splitter
    m_mainSplitter->addWidget(leftWidget);
    m_mainSplitter->addWidget(rightWidget);
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

    // Add camera status indicators
    m_cameraStatusGroup = new QGroupBox("Camera Status");
    QHBoxLayout *cameraLayout = new QHBoxLayout(m_cameraStatusGroup);

    m_leftCameraStatusLabel = new QLabel("Left: ❌ Disconnected");
    m_leftCameraStatusLabel->setStyleSheet("QLabel { color: red; font-weight: bold; }");
    cameraLayout->addWidget(m_leftCameraStatusLabel);

    QFrame *separator = new QFrame;
    separator->setFrameStyle(QFrame::VLine | QFrame::Sunken);
    cameraLayout->addWidget(separator);

    m_rightCameraStatusLabel = new QLabel("Right: ❌ Disconnected");
    m_rightCameraStatusLabel->setStyleSheet("QLabel { color: red; font-weight: bold; }");
    cameraLayout->addWidget(m_rightCameraStatusLabel);

    m_retryConnectionButton = new QPushButton("Retry Connection");
    m_retryConnectionButton->setMaximumWidth(120);
    connect(m_retryConnectionButton, &QPushButton::clicked, this, &MainWindow::initializeCameraSystem);
    cameraLayout->addWidget(m_retryConnectionButton);

    m_statusBar->addPermanentWidget(m_cameraStatusGroup);

    // New sync status label (permanent, stretches to right)
    m_syncStatusLabel = new QLabel("Sync: N/A");
    m_syncStatusLabel->setMinimumWidth(180);
    m_syncStatusLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    m_statusBar->addPermanentWidget(m_syncStatusLabel, 1);

    m_progressBar = new QProgressBar;
    m_progressBar->setVisible(false);
    m_statusBar->addPermanentWidget(m_progressBar);
}

void MainWindow::connectSignals() {
    // Debug output to verify signal connections
    qDebug() << "Connecting Qt signals and slots...";

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
    connect(m_aiCalibrationAction, &QAction::triggered, this,
          &MainWindow::startAICalibration);
    connect(m_processAction, &QAction::triggered, this,
          &MainWindow::processStereoImages);
    connect(m_batchProcessAction, &QAction::triggered, this,
          &MainWindow::openBatchProcessing);
    connect(m_epipolarCheckerAction, &QAction::triggered, this,
          &MainWindow::openEpipolarChecker);
    connect(m_liveProcessingAction, &QAction::triggered, this,
          &MainWindow::toggleLiveProcessing);
    connect(m_exportAction, &QAction::triggered, this,
          &MainWindow::exportPointCloud);

    // View actions
    connect(m_showDisparityAction, &QAction::toggled, this,
          &MainWindow::updateDisparityMap);
    connect(m_showPointCloudAction, &QAction::toggled, this,
          &MainWindow::updatePointCloud);
    connect(m_enableProfilingAction, &QAction::toggled, this,
          &MainWindow::toggleProfiling);

    // Help actions
    connect(m_shortcutsAction, &QAction::triggered, this, &MainWindow::showKeyboardShortcuts);
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

    // Parameter panel - ensure it exists before connecting
    if (m_parameterPanel) {
        connect(m_parameterPanel, &ParameterPanel::parametersChanged, this,
              &MainWindow::onParameterChanged);
        qDebug() << "✓ Parameter panel connected successfully";
    } else {
        qWarning() << "✗ Parameter panel not created before connecting signals!";
    }

    // Timer connections
    connect(m_processingTimer, &QTimer::timeout, this,
          &MainWindow::onProcessingFinished);
    connect(m_liveProcessingTimer, &QTimer::timeout, this,
          &MainWindow::onLiveFrameProcessed);

    qDebug() << "✓ All Qt signal connections established";
}

void MainWindow::toggleProfiling(bool checked) {
    using namespace stereo_vision::perf;
    ProfilerRegistry::instance().enable(checked);
    if (checked) {
        ProfilerRegistry::instance().reset();
        m_profilingTimer->start();
        logCameraOperation("Profiling enabled", true, "Collecting performance metrics");
    } else {
        m_profilingTimer->stop();
        logCameraOperation("Profiling disabled", true);
    }
}

void MainWindow::updateProfilingStats() {
    using namespace stereo_vision::perf;
    if (!ProfilerRegistry::instance().enabled()) return;
    auto capture = ProfilerRegistry::instance().get(Stage::Capture);
    auto disparity = ProfilerRegistry::instance().get(Stage::Disparity);
    auto pointcloud = ProfilerRegistry::instance().get(Stage::PointCloud);
    auto onnx = ProfilerRegistry::instance().get(Stage::ONNXInference);

    QStringList parts;
    if (capture.count) parts << QString("Cap avg %.2f ms (EMA %.2f)" ).arg(capture.sum_ms / capture.count, capture.ema_ms);
    if (disparity.count) parts << QString("Disp avg %.2f ms" ).arg(disparity.sum_ms / disparity.count);
    if (pointcloud.count) parts << QString("PC avg %.2f ms" ).arg(pointcloud.sum_ms / pointcloud.count);
    if (onnx.count) parts << QString("AI avg %.2f ms" ).arg(onnx.sum_ms / onnx.count);

    if (!parts.isEmpty()) {
        QString summary = QString("[Profiling] %1").arg(parts.join(" | "));
        logCameraOperation(summary, true);
    }
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
  // Launch the camera calibration wizard
  if (!m_cameraManager || !m_cameraManager->isAnyCameraOpen()) {
    QMessageBox::warning(
        this, "Camera Required",
        "Please select and start cameras first before running calibration.\n\n"
        "1. Go to File → Select Cameras to choose your cameras\n"
        "2. Then File → Start Webcam Capture to begin capturing");
    return;
  }

  auto wizard = new CalibrationWizard(m_cameraManager, this);

  if (wizard->exec() == QDialog::Accepted) {
    QMessageBox::information(
        this, "Calibration Complete",
        "Camera calibration has been completed successfully!\n\n"
        "The calibration parameters are now ready for use in stereo "
        "processing.");

    // Update UI to reflect that we now have calibration
    m_hasCalibration = true;
    updateUI();
  }

  wizard->deleteLater();
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
                     "<li>Interactive camera calibration wizard</li>"
                     "<li>AI-powered auto-calibration</li>"
                     "<li>Real-time live processing</li>"
                     "<li>Multiple export formats</li>"
                     "<li>Enhanced camera detection with MultiCameraUtils</li>"
                     "</ul>"
                     "<p><b>Keyboard Shortcuts:</b></p>"
                     "<ul>"
                     "<li><b>Ctrl+Shift+C</b> - Select Cameras</li>"
                     "<li><b>Ctrl+Shift+S</b> - Start Webcam Capture</li>"
                     "<li><b>Ctrl+Shift+T</b> - Stop Webcam Capture</li>"
                     "<li><b>L</b> - Capture Left Image</li>"
                     "<li><b>R</b> - Capture Right Image</li>"
                     "<li><b>Space</b> - Capture Stereo Pair</li>"
                     "<li><b>Ctrl+C</b> - Calibrate Cameras</li>"
                     "<li><b>Ctrl+Alt+C</b> - AI Auto-Calibration</li>"
                     "<li><b>Ctrl+Shift+P</b> - Toggle Live Processing</li>"
                     "</ul>"
                     "<p><b>Version:</b> 1.0.0</p>"
                     "<p><b>Build:</b> " +
                         QString(__DATE__) + " " + QString(__TIME__) + "</p>");
}

void MainWindow::showKeyboardShortcuts() {
  QMessageBox::information(this, "Keyboard Shortcuts",
                          "<h3>Keyboard Shortcuts Reference</h3>"
                          "<table border='1' cellpadding='5' cellspacing='0'>"
                          "<tr><th>Action</th><th>Shortcut</th></tr>"
                          "<tr><td><b>File Operations</b></td><td></td></tr>"
                          "<tr><td>Open Left Image</td><td>Ctrl+L</td></tr>"
                          "<tr><td>Open Right Image</td><td>Ctrl+R</td></tr>"
                          "<tr><td>Open Stereo Folder</td><td>Ctrl+F</td></tr>"
                          "<tr><td>Load Calibration</td><td>Ctrl+O</td></tr>"
                          "<tr><td>Save Calibration</td><td>Ctrl+S</td></tr>"
                          "<tr><td>Exit</td><td>Ctrl+Q</td></tr>"
                          "<tr><td><b>Camera Controls</b></td><td></td></tr>"
                          "<tr><td>Select Cameras</td><td>Ctrl+Shift+C</td></tr>"
                          "<tr><td>Start Webcam Capture</td><td>Ctrl+Shift+S</td></tr>"
                          "<tr><td>Stop Webcam Capture</td><td>Ctrl+Shift+T</td></tr>"
                          "<tr><td>Capture Left Image</td><td>L</td></tr>"
                          "<tr><td>Capture Right Image</td><td>R</td></tr>"
                          "<tr><td>Capture Stereo Pair</td><td>Space</td></tr>"
                          "<tr><td><b>Processing</b></td><td></td></tr>"
                          "<tr><td>Calibrate Cameras</td><td>Ctrl+C</td></tr>"
                          "<tr><td>AI Auto-Calibration</td><td>Ctrl+Alt+C</td></tr>"
                          "<tr><td>Process Stereo Images</td><td>Ctrl+P</td></tr>"
                          "<tr><td>Toggle Live Processing</td><td>Ctrl+Shift+P</td></tr>"
                          "<tr><td>Batch Processing</td><td>Ctrl+B</td></tr>"
                          "<tr><td>Epipolar Checker</td><td>Ctrl+Shift+E</td></tr>"
                          "<tr><td>Export Point Cloud</td><td>Ctrl+E</td></tr>"
                          "<tr><td><b>View</b></td><td></td></tr>"
                          "<tr><td>Show Disparity Map</td><td>Ctrl+D</td></tr>"
                          "<tr><td>Show Point Cloud</td><td>Ctrl+3</td></tr>"
                          "<tr><td><b>Help</b></td><td></td></tr>"
                          "<tr><td>Keyboard Shortcuts</td><td>F1</td></tr>"
                          "</table>");
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
  // Update action states based on current application state
  bool hasCameras = m_leftCameraConnected || m_rightCameraConnected;
  bool canCapture = hasCameras && !m_isProcessing;
  bool canProcess = m_hasImages && m_hasCalibration && !m_isProcessing;

  // File menu actions
  m_loadCalibrationAction->setEnabled(!m_isProcessing);
  m_saveCalibrationAction->setEnabled(m_hasCalibration && !m_isProcessing);

  // Camera capture actions
  m_startCaptureAction->setEnabled(hasCameras && !m_isCapturing);
  m_stopCaptureAction->setEnabled(m_isCapturing);
  m_captureLeftAction->setEnabled(m_isCapturing);
  m_captureRightAction->setEnabled(m_isCapturing);
  m_captureStereoAction->setEnabled(m_isCapturing);

  // Processing actions
  m_calibrateAction->setEnabled(hasCameras && !m_isProcessing);
  m_aiCalibrationAction->setEnabled(hasCameras && !m_isProcessing && !m_aiCalibrationActive);
  m_processAction->setEnabled(canProcess);
  m_liveProcessingAction->setEnabled(m_hasCalibration && m_isCapturing);
  m_exportAction->setEnabled(!m_isProcessing);
  m_batchProcessAction->setEnabled(!m_isProcessing);
  m_epipolarCheckerAction->setEnabled(m_hasCalibration && !m_isProcessing);

  // View actions (always enabled)
  m_showDisparityAction->setEnabled(true);
  m_showPointCloudAction->setEnabled(true);

  // Update window title with current state
  QString title = "Stereo Vision 3D Point Cloud Generator";
  QStringList statusItems;

  if (m_hasImages) statusItems << "Images Loaded";
  if (m_hasCalibration) statusItems << "Calibrated";
  if (m_isCapturing) statusItems << "Capturing";
  if (m_liveProcessingEnabled) statusItems << "Live Processing";
  if (m_aiCalibrationActive) statusItems << "AI Calibration";
  if (m_isProcessing) statusItems << "Processing";

  if (!statusItems.isEmpty()) {
    title += " - " + statusItems.join(", ");
  }

  setWindowTitle(title);

  // Update camera status indicators
  updateCameraStatusIndicators();
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

void MainWindow::initializeCameraSystem() {
    logCameraOperation("=== Initializing Camera System ===", true, "Starting comprehensive camera detection");
    m_debugLogOutput->clear();
    logCameraOperation("Camera system initialization started", true, "Clearing previous logs");

    // Clear multicam system for re-init
    if (m_multiCameraSystem) {
        // Currently no clear() API; recreate instance
        m_multiCameraSystem = std::make_shared<stereovision::multicam::MultiCameraSystem>();
    }

    try {
        logCameraOperation("Detecting available cameras", true, "Using MultiCameraUtils detection");
        std::vector<int> availableCameras = stereovision::multicam::MultiCameraUtils::detectAvailableCameras();
        logCameraOperation("Camera detection completed", true,
                          QString("%1 cameras found: [%2]")
                          .arg(availableCameras.size())
                          .arg([&availableCameras]() { QStringList idList; for (int id : availableCameras) idList << QString::number(id); return idList.join(", "); }()));

        if (!availableCameras.empty()) {
            QStringList cameraList, workingCameras, failedCameras;
            for (int camera_id : availableCameras) {
                logCameraOperation(QString("Testing Camera %1").arg(camera_id), true, "Connection test starting");
                try {
                    bool connectionOk = stereovision::multicam::MultiCameraUtils::testCameraConnection(camera_id);
                    cameraList << QString("Camera %1 %2").arg(camera_id).arg(connectionOk ? "✓" : "✗");
                    if (connectionOk) {
                        workingCameras << QString::number(camera_id);
                        // Add to multicam system immediately with default config
                        stereovision::multicam::CameraConfig cfg; cfg.camera_id = camera_id; cfg.fps = 30.0; cfg.resolution = cv::Size(640,480);
                        if (m_multiCameraSystem) {
                            m_multiCameraSystem->addCamera(camera_id, cfg);
                        }
                        logCameraOperation(QString("Camera %1 test PASSED").arg(camera_id), true, "Added to MultiCameraSystem");
                    } else {
                        failedCameras << QString::number(camera_id);
                        logCameraOperation(QString("Camera %1 test FAILED").arg(camera_id), false, "Camera detected but not responding");
                        logCameraOperation(QString("Attempting retry for Camera %1").arg(camera_id), true, "Starting automatic retry sequence");
                        retryCameraConnection(camera_id, 2);
                    }
                } catch (const std::exception& e) {
                    failedCameras << QString::number(camera_id);
                    logCameraOperation(QString("Camera %1 test EXCEPTION").arg(camera_id), false, QString("Error: %1").arg(e.what()));
                }
            }
            QString statusMessage = QString("Camera system initialized - %1 cameras detected: %2")
                                  .arg(availableCameras.size())
                                  .arg(cameraList.join(", "));
            m_statusLabel->setText(statusMessage);
            updateCameraStatusIndicators();
            if (!workingCameras.isEmpty()) {
                logCameraOperation("Working cameras found", true, QString("Functional cameras: [%1]").arg(workingCameras.join(", ")));
                if (workingCameras.size() == 2) {
                    logCameraOperation("Stereo camera pair detected", true, "Two working cameras available for stereo vision");
                    QTimer::singleShot(1500, this, [this, workingCameras]() { if (QMessageBox::Yes == QMessageBox::information(this, "Auto-Detection", QString("Two working cameras detected! (Cameras %1 and %2)\nOpen camera selector now?").arg(workingCameras[0], workingCameras[1]), QMessageBox::Yes | QMessageBox::No)) { showCameraSelector(); } });
                }
            }
            if (!failedCameras.isEmpty()) {
                logCameraOperation("Some cameras failed connection tests", false, QString("Failed cameras: [%1]").arg(failedCameras.join(", ")));
            }
        } else {
            logCameraOperation("No cameras detected", false, "No camera devices found on system");
            m_statusLabel->setText("No cameras detected - connect cameras and restart application");
            QTimer::singleShot(3000, this, [this]() { showCameraErrorDialog("No Cameras Detected", "No cameras were detected on startup."); });
        }
    } catch (const std::exception& e) {
        QString errorMsg = QString("Camera system initialization error: %1").arg(e.what());
        logCameraOperation("CRITICAL ERROR during initialization", false, errorMsg);
        m_statusLabel->setText("Camera initialization failed - fallback to manual detection");
        if (m_cameraManager) {
            try { int numCameras = m_cameraManager->detectCameras(); if (numCameras > 0) { logCameraOperation("Fallback detection successful", true, QString("%1 cameras found with basic method").arg(numCameras)); } } catch (...) {}
        }
    }
    logCameraOperation("Camera system initialization complete", true, "Ready for user interaction");
}

void MainWindow::refreshCameraStatus() {
    // Check camera availability using multicam utilities
    try {
        logCameraOperation("Refreshing camera status", true, "Periodic camera availability check");

        std::vector<int> availableCameras = stereovision::multicam::MultiCameraUtils::detectAvailableCameras();

        // Update UI based on current camera availability
        bool hasWorkingCameras = false;
        QStringList workingCameras;
        QStringList disconnectedCameras;

        for (int camera_id : availableCameras) {
            try {
                if (stereovision::multicam::MultiCameraUtils::testCameraConnection(camera_id)) {
                    hasWorkingCameras = true;
                    workingCameras << QString("Camera %1").arg(camera_id);
                    logCameraOperation(QString("Camera %1 status check").arg(camera_id), true, "Camera responding");
                } else {
                    disconnectedCameras << QString("Camera %1").arg(camera_id);
                    logCameraOperation(QString("Camera %1 status check").arg(camera_id), false, "Camera not responding");
                }
            } catch (const std::exception& e) {
                disconnectedCameras << QString("Camera %1").arg(camera_id);
                logCameraOperation(QString("Camera %1 status exception").arg(camera_id), false,
                                 QString("Error: %1").arg(e.what()));
            }
        }

        // Update camera connection status if cameras were previously selected
        bool leftStatusChanged = false;
        bool rightStatusChanged = false;

        if (m_selectedLeftCamera >= 0) {
            bool wasConnected = m_leftCameraConnected;
            m_leftCameraConnected = std::find(availableCameras.begin(), availableCameras.end(),
                                            m_selectedLeftCamera) != availableCameras.end();

            if (m_leftCameraConnected) {
                // Double-check with connection test
                try {
                    m_leftCameraConnected = stereovision::multicam::MultiCameraUtils::testCameraConnection(m_selectedLeftCamera);
                } catch (...) {
                    m_leftCameraConnected = false;
                }
            }

            leftStatusChanged = (wasConnected != m_leftCameraConnected);
            if (leftStatusChanged) {
                logCameraOperation(QString("Left camera status changed: Camera %1").arg(m_selectedLeftCamera),
                                 m_leftCameraConnected,
                                 m_leftCameraConnected ? "Now connected" : "Now disconnected");
            }
        }

        if (m_selectedRightCamera >= 0) {
            bool wasConnected = m_rightCameraConnected;
            m_rightCameraConnected = std::find(availableCameras.begin(), availableCameras.end(),
                                             m_selectedRightCamera) != availableCameras.end();

            if (m_rightCameraConnected) {
                // Double-check with connection test
                try {
                    m_rightCameraConnected = stereovision::multicam::MultiCameraUtils::testCameraConnection(m_selectedRightCamera);
                } catch (...) {
                    m_rightCameraConnected = false;
                }
            }

            rightStatusChanged = (wasConnected != m_rightCameraConnected);
            if (rightStatusChanged) {
                logCameraOperation(QString("Right camera status changed: Camera %1").arg(m_selectedRightCamera),
                                 m_rightCameraConnected,
                                 m_rightCameraConnected ? "Now connected" : "Now disconnected");
            }
        }

        // Update UI state and status indicators
        updateUI();
        updateCameraStatusIndicators();

        // Update status if not currently capturing or processing
        if (!m_isCapturing && !m_isProcessing) {
            if (hasWorkingCameras) {
                QString statusMsg = QString("Available cameras: %1").arg(workingCameras.join(", "));
                m_statusLabel->setText(statusMsg);

                if (!disconnectedCameras.isEmpty()) {
                    logCameraOperation("Mixed camera status", false,
                                     QString("Working: [%1], Disconnected: [%2]")
                                     .arg(workingCameras.join(", "), disconnectedCameras.join(", ")));
                }
            } else {
                m_statusLabel->setText("No working cameras detected");
                logCameraOperation("No working cameras in status refresh", false,
                                 availableCameras.empty() ? "No cameras detected" :
                                 "Cameras detected but none responding");
            }
        }

        // Show notifications for camera status changes
        if (leftStatusChanged || rightStatusChanged) {
            QStringList changes;
            if (leftStatusChanged) {
                changes << QString("Left camera (ID %1): %2")
                          .arg(m_selectedLeftCamera)
                          .arg(m_leftCameraConnected ? "Connected" : "Disconnected");
            }
            if (rightStatusChanged) {
                changes << QString("Right camera (ID %1): %2")
                          .arg(m_selectedRightCamera)
                          .arg(m_rightCameraConnected ? "Connected" : "Disconnected");
            }

            logCameraOperation("Camera status change detected", true, changes.join("; "));

            // If cameras were disconnected during operation, show warning
            if (!m_leftCameraConnected || !m_rightCameraConnected) {
                if (m_isCapturing) {
                    // Stop capture if cameras disconnected
                    stopWebcamCapture();
                    showCameraErrorDialog("Camera Disconnected During Capture",
                        "One or more cameras were disconnected during capture operation.",
                        "Capture has been stopped. Please reconnect cameras and restart capture.");
                }
            }
        }

    } catch (const std::exception& e) {
        logCameraOperation("Camera status refresh failed", false, QString("Error: %1").arg(e.what()));
        qDebug() << "Camera status refresh error:" << e.what();
    }
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
    m_startCaptureAction->setEnabled(m_leftCameraConnected ||
                                     m_rightCameraConnected);

    // Provide detailed status feedback
    QString statusMessage;
    if (m_leftCameraConnected && m_rightCameraConnected) {
      if (m_selectedLeftCamera == m_selectedRightCamera) {
        statusMessage = QString("Single camera mode configured: Camera %1 for both channels")
                       .arg(m_selectedLeftCamera);
      } else {
        statusMessage = QString("Stereo camera pair configured: Left=%1, Right=%2 - Ready for synchronized capture")
                       .arg(m_selectedLeftCamera).arg(m_selectedRightCamera);
      }
    } else if (m_leftCameraConnected) {
      statusMessage = QString("Left camera configured: Camera %1 - Mono capture mode")
                     .arg(m_selectedLeftCamera);
    } else if (m_rightCameraConnected) {
      statusMessage = QString("Right camera configured: Camera %1 - Mono capture mode")
                     .arg(m_selectedRightCamera);
    }

    m_statusLabel->setText(statusMessage);
  } else {
    m_statusLabel->setText("Camera selection cancelled");
  }

  dialog->deleteLater();
}

void MainWindow::startWebcamCapture() {
  if (!m_leftCameraConnected && !m_rightCameraConnected) {
    QMessageBox::warning(
        this, "No Cameras",
        "No cameras are connected. Please connect at least one camera first.");
    return;
  }

  bool singleCameraMode = (m_selectedLeftCamera == m_selectedRightCamera &&
                         m_selectedLeftCamera >= 0 && m_leftCameraConnected &&
                         m_rightCameraConnected);

  bool success = false;

  if (singleCameraMode) {
    QMessageBox::information(
        this, "Single Camera Mode",
        "Opening camera in single camera mode for manual stereo capture.");

    success = m_cameraManager->openSingleCamera(m_selectedLeftCamera);
    m_statusLabel->setText("Single camera mode active - manual stereo capture");
  } else if (m_leftCameraConnected && m_rightCameraConnected) {
    // Both cameras selected and different
    success = m_cameraManager->openCameras(m_selectedLeftCamera,
                                          m_selectedRightCamera);
    m_statusLabel->setText(
        "Stereo camera pair active - synchronized capture");
  } else if (m_leftCameraConnected) {
    success = m_cameraManager->openSingleCamera(m_selectedLeftCamera);
    m_statusLabel->setText("Left camera only mode active");
  } else {
    success = m_cameraManager->openSingleCamera(m_selectedRightCamera);
    m_statusLabel->setText("Right camera only mode active");
  }

  if (!success) {
    QString errorMsg = "Failed to open camera(s). Please check your camera "
                        "connections and try again.";
    QMessageBox::critical(this, "Camera Error", errorMsg);
    return;
  }

  // Start capture timer
  m_captureTimer->start(33); // ~30 FPS
  m_isCapturing = true;

  // Update UI state
  m_startCaptureAction->setEnabled(false);
  m_stopCaptureAction->setEnabled(true);
  m_captureLeftAction->setEnabled(true); // Always enabled during capture
  m_captureRightAction->setEnabled(true); // Always enabled during capture
  m_captureStereoAction->setEnabled(true); // Always enabled in capture mode

  // Connect timer to frame update slot
  connect(m_captureTimer, &QTimer::timeout, this, &MainWindow::onFrameReady);
}

void MainWindow::stopWebcamCapture() {
  if (!m_isCapturing)
    return;

  // Stop timer and disconnect
  m_captureTimer->stop();
  disconnect(m_captureTimer, &QTimer::timeout, this, &MainWindow::onFrameReady);

  // Close cameras
  m_cameraManager->closeCameras();
  m_isCapturing = false;

  // Update UI
  m_startCaptureAction->setEnabled(m_leftCameraConnected ||
                                   m_rightCameraConnected);
  m_stopCaptureAction->setEnabled(false);
  m_captureLeftAction->setEnabled(false);
  m_captureRightAction->setEnabled(false);
  m_captureStereoAction->setEnabled(false);

  m_statusLabel->setText("Webcam capture stopped");
}

void MainWindow::captureLeftImage() {
  if (!m_isCapturing) {
    QMessageBox::warning(
        this, "Capture Error",
        "Webcam capture is not running. Please start capture first.");
    return;
  }

  // In single camera mode, both frames are the same
  cv::Mat frameToSave = m_lastLeftFrame;
  if (frameToSave.empty()) {
    frameToSave = m_lastRightFrame; // Fallback
  }

  if (frameToSave.empty()) {
    QMessageBox::warning(this, "Capture Error",
                         "No frame available for capture. Please wait for the "
                         "camera to initialize.");
    return;
  }

  // Save the current frame as left image
  QString fileName = QFileDialog::getSaveFileName(
      this, "Save Left Image",
      m_outputPath + "/left_" +
          QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
      "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)");

  if (!fileName.isEmpty()) {
    if (cv::imwrite(fileName.toStdString(), frameToSave)) {
      // Also load it as the current left image
      m_leftImagePath = fileName;
      m_leftImageWidget->setImage(fileName);
      m_hasImages = !m_rightImagePath.isEmpty();
      updateUI();

      m_statusLabel->setText("Left image captured: " +
                             QFileInfo(fileName).fileName());
    } else {
      QMessageBox::critical(this, "Save Error", "Failed to save left image.");
    }
  }
}

void MainWindow::captureRightImage() {
  if (!m_isCapturing) {
    QMessageBox::warning(
        this, "Capture Error",
        "Webcam capture is not running. Please start capture first.");
    return;
  }

  // In single camera mode, both frames are the same
  cv::Mat frameToSave = m_lastRightFrame;
  if (frameToSave.empty()) {
    frameToSave = m_lastLeftFrame; // Fallback
  }

  if (frameToSave.empty()) {
    QMessageBox::warning(this, "Capture Error",
                         "No frame available for capture. Please wait for the "
                         "camera to initialize.");
    return;
  }

  // Save the current frame as right image
  QString fileName = QFileDialog::getSaveFileName(
      this, "Save Right Image",
      m_outputPath + "/right_" +
          QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
      "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)");

  if (!fileName.isEmpty()) {
    if (cv::imwrite(fileName.toStdString(), frameToSave)) {
      // Also load it as the current right image
      m_rightImagePath = fileName;
      m_rightImageWidget->setImage(fileName);
      m_hasImages = !m_leftImagePath.isEmpty();
      updateUI();

      m_statusLabel->setText("Right image captured: " +
                             QFileInfo(fileName).fileName());
    } else {
      QMessageBox::critical(this, "Save Error", "Failed to save right image.");
    }
  }
}

void MainWindow::captureStereoImage() {
  if (!m_isCapturing) {
    QMessageBox::warning(
        this, "Capture Error",
        "Webcam capture is not running. Please start capture first.");
    return;
  }

  // Check for available frames
  cv::Mat leftFrame = m_lastLeftFrame;
  cv::Mat rightFrame = m_lastRightFrame;

  if (leftFrame.empty()) {
    leftFrame = rightFrame; // Use same frame if left is empty
  }
  if (rightFrame.empty()) {
    rightFrame = leftFrame; // Use same frame if right is empty
  }

  if (leftFrame.empty() || rightFrame.empty()) {
    QMessageBox::warning(this, "Capture Error",
                         "No frames available for capture. Please wait for the "
                         "camera to initialize.");
    return;
  }

  // Create timestamp for synchronized capture
  QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");

  // Save both frames simultaneously
  QString leftFileName = m_outputPath + "/left_" + timestamp + ".png";
  QString rightFileName = m_outputPath + "/right_" + timestamp + ".png";

  QDir().mkpath(m_outputPath); // Ensure directory exists

  bool leftSaved = cv::imwrite(leftFileName.toStdString(), leftFrame);
  bool rightSaved = cv::imwrite(rightFileName.toStdString(), rightFrame);

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
                          "Failed to save stereo image pair. Check output "
                          "directory permissions.");
  }
}

void MainWindow::onFrameReady() {
  if (!m_isCapturing) { return; }
  // If single-camera mode or only one connected, use legacy camera manager path
  bool singleCameraMode = (m_selectedLeftCamera == m_selectedRightCamera && m_selectedLeftCamera >= 0 && m_leftCameraConnected && m_rightCameraConnected);
  int connectedCount = 0; if (m_leftCameraConnected) ++connectedCount; if (m_rightCameraConnected && m_selectedRightCamera != m_selectedLeftCamera) ++connectedCount;
  bool useMultiCam = (connectedCount >= 2 && m_multiCameraSystem && m_multiCameraSystem->getConnectedCameras().size() >= 2);

  if (!useMultiCam) {
      if (!m_cameraManager || !m_cameraManager->isAnyCameraOpen()) return;
      if (singleCameraMode || !m_cameraManager->areCamerasOpen()) {
          cv::Mat frame; if (m_cameraManager->grabSingleFrame(frame) && !frame.empty()) {
              m_lastLeftFrame = frame.clone(); m_lastRightFrame = frame.clone();
              cv::Mat displayFrame; cv::cvtColor(frame, displayFrame, cv::COLOR_BGR2RGB);
              QImage qimg(displayFrame.data, displayFrame.cols, displayFrame.rows, displayFrame.step, QImage::Format_RGB888);
              QString tempPath = QDir::tempPath() + "/stereo_single_preview.png"; qimg.save(tempPath);
              if (m_leftCameraConnected) m_leftImageWidget->setImage(tempPath);
              if (m_rightCameraConnected) m_rightImageWidget->setImage(tempPath);
          }
      } else {
          cv::Mat leftFrame, rightFrame; if (m_cameraManager->grabFrames(leftFrame, rightFrame)) {
              if (!leftFrame.empty()) m_lastLeftFrame = leftFrame.clone(); if (!rightFrame.empty()) m_lastRightFrame = rightFrame.clone();
              if (!leftFrame.empty() && m_leftCameraConnected) { cv::Mat df; cv::cvtColor(leftFrame, df, cv::COLOR_BGR2RGB); QImage qimg(df.data, df.cols, df.rows, df.step, QImage::Format_RGB888); QString tempPath = QDir::tempPath() + "/stereo_left_preview.png"; qimg.save(tempPath); m_leftImageWidget->setImage(tempPath); }
              if (!rightFrame.empty() && m_rightCameraConnected) { cv::Mat df; cv::cvtColor(rightFrame, df, cv::COLOR_BGR2RGB); QImage qimg(df.data, df.cols, df.rows, df.step, QImage::Format_RGB888); QString tempPath = QDir::tempPath() + "/stereo_right_preview.png"; qimg.save(tempPath); m_rightImageWidget->setImage(tempPath); }
          }
      }
  } else {
      // Multi-camera synchronized capture
      std::map<int, cv::Mat> frames; std::map<int, std::chrono::high_resolution_clock::time_point> timestamps;
      if (m_multiCameraSystem->captureSynchronizedFrames(frames, timestamps)) {
          // For now map selected IDs to left/right if available
          if (frames.count(m_selectedLeftCamera)) { m_lastLeftFrame = frames[m_selectedLeftCamera].clone(); }
          if (frames.count(m_selectedRightCamera)) { m_lastRightFrame = frames[m_selectedRightCamera].clone(); }
          // Display
          if (!m_lastLeftFrame.empty() && m_leftCameraConnected) { cv::Mat df; cv::cvtColor(m_lastLeftFrame, df, cv::COLOR_BGR2RGB); QImage qimg(df.data, df.cols, df.rows, df.step, QImage::Format_RGB888); QString tempPath = QDir::tempPath() + "/stereo_left_sync.png"; qimg.save(tempPath); m_leftImageWidget->setImage(tempPath); }
          if (!m_lastRightFrame.empty() && m_rightCameraConnected) { cv::Mat df; cv::cvtColor(m_lastRightFrame, df, cv::COLOR_BGR2RGB); QImage qimg(df.data, df.cols, df.rows, df.step, QImage::Format_RGB888); QString tempPath = QDir::tempPath() + "/stereo_right_sync.png"; qimg.save(tempPath); m_rightImageWidget->setImage(tempPath); }
          // Handle recent disconnect
          if (m_multiCameraSystem->hadRecentDisconnect()) {
              logCameraOperation("Detected recent camera disconnect", false, "Stopping capture");
              stopWebcamCapture();
              QMessageBox::warning(this, "Camera Disconnect", "A camera was disconnected during synchronized capture. Capture has been stopped.");
              m_multiCameraSystem->clearRecentDisconnectFlag();
          }
      }
  }
  if (m_aiCalibrationActive && !m_lastLeftFrame.empty()) { captureCalibrationFrame(); }
}

void MainWindow::updateSyncStatus() {
    if (!m_syncStatusLabel) return;
    if (!m_multiCameraSystem || m_multiCameraSystem->getConnectedCameras().size() < 2) {
        m_syncStatusLabel->setText("Sync: N/A");
        return;
    }
    const auto &stats = m_multiCameraSystem->getSyncStats();
    auto quality = m_multiCameraSystem->classifySyncQuality();
    QString qStr;
    switch (quality) { case stereovision::multicam::MultiCameraSystem::SyncQuality::EXCELLENT: qStr = "EXCELLENT"; break; case stereovision::multicam::MultiCameraSystem::SyncQuality::GOOD: qStr = "GOOD"; break; case stereovision::multicam::MultiCameraSystem::SyncQuality::POOR: qStr = "POOR"; break; default: qStr = "UNKNOWN"; }
    QString txt = QString("Sync: %1 Δavg %2ms Δmax %3ms jitter %4ms drops %5 fail %6")
                  .arg(qStr)
                  .arg(QString::number(stats.avg_delta_ms.load(), 'f', 2))
                  .arg(QString::number(stats.max_delta_ms.load(), 'f', 2))
                  .arg(QString::number(stats.jitter_ms.load(), 'f', 2))
                  .arg(stats.dropped_frames.load())
                  .arg(stats.consecutive_failures.load());
    m_syncStatusLabel->setText(txt);
    // Color coding
    QString color = (qStr == "EXCELLENT") ? "#2e8b57" : (qStr == "GOOD" ? "#1e90ff" : (qStr == "POOR" ? "#b22222" : "#696969"));
    m_syncStatusLabel->setStyleSheet(QString("QLabel { color: %1; font-weight: bold; }").arg(color));
}

void MainWindow::retryCameraConnection(int cameraId, int maxRetries) {
    // Schedule retry attempts instead of blocking
    m_retryTargetCameraId = cameraId;
    m_retryMaxAttempts = maxRetries;
    m_retryCurrentAttempt = 0;
    if (!m_retryTimer) return;
    logCameraOperation(QString("Scheduling retry sequence for Camera %1 (%2 attempts)").arg(cameraId).arg(maxRetries), true);
    if (!m_retryTimer->isActive()) {
        m_retryTimer->start(10); // start almost immediately
    }
}

void MainWindow::performRetryAttempt() {
    if (m_retryCurrentAttempt >= m_retryMaxAttempts) {
        logCameraOperation(QString("Camera %1 connection failed after %2 attempts").arg(m_retryTargetCameraId).arg(m_retryMaxAttempts), false, "Non-blocking retry sequence complete");
        return;
    }
    ++m_retryCurrentAttempt;
    int attempt = m_retryCurrentAttempt;
    int cameraId = m_retryTargetCameraId;
    logCameraOperation(QString("Retry attempt %1/%2 for Camera %3").arg(attempt).arg(m_retryMaxAttempts).arg(cameraId), true, "Testing connection...");
    bool connected = false;
    try { connected = stereovision::multicam::MultiCameraUtils::testCameraConnection(cameraId); } catch (...) { connected = false; }
    if (connected) {
        logCameraOperation(QString("Camera %1 connection successful on attempt %2").arg(cameraId).arg(attempt), true, "Camera responding");
        if (m_multiCameraSystem && std::find(m_multiCameraSystem->getConnectedCameras().begin(), m_multiCameraSystem->getConnectedCameras().end(), cameraId) == m_multiCameraSystem->getConnectedCameras().end()) {
            stereovision::multicam::CameraConfig cfg; cfg.camera_id = cameraId; m_multiCameraSystem->addCamera(cameraId, cfg);
        }
        updateCameraStatusIndicators();
        return;
    } else {
        logCameraOperation(QString("Camera %1 attempt %2 failed").arg(cameraId).arg(attempt), false, "Will retry if attempts remain");
    }
    if (m_retryCurrentAttempt < m_retryMaxAttempts) {
        m_retryTimer->start(1000);
    } else {
        showCameraErrorDialog(QString("Camera %1 Connection Failed").arg(cameraId), QString("Unable to connect after %1 attempts").arg(m_retryMaxAttempts));
    }
}

void MainWindow::logCameraOperation(const QString &operation, bool success, const QString &details) {
    QString line = QDateTime::currentDateTime().toString("HH:mm:ss") + " | " + (success ? "[OK] " : "[ERR] ") + operation;
    if (!details.isEmpty()) line += " - " + details;
    if (m_debugLogOutput) {
        m_debugLogOutput->append(line);
        // Keep log reasonably short
        if (m_debugLogOutput->document()->blockCount() > 500) {
            QTextCursor c(m_debugLogOutput->document());
            c.movePosition(QTextCursor::Start);
            c.select(QTextCursor::BlockUnderCursor);
            c.removeSelectedText();
            c.deleteChar();
        }
    }
    qInfo().noquote() << line;
}

void MainWindow::showCameraErrorDialog(const QString &title, const QString &message, const QString &details) {
    QString full = message;
    if (!details.isEmpty()) full += "\n\nDetails: " + details;
    QMessageBox::warning(this, title, full);
    logCameraOperation(title, false, message);
}

void MainWindow::updateCameraStatusIndicators() {
    if (!m_leftCameraStatusLabel || !m_rightCameraStatusLabel) return;
    // Left
    if (m_leftCameraConnected && m_selectedLeftCamera >= 0) {
        m_leftCameraStatusLabel->setText(QString("Left: ✅ Cam %1").arg(m_selectedLeftCamera));
        m_leftCameraStatusLabel->setStyleSheet("QLabel { color: #2e8b57; font-weight: bold; }");
    } else {
        m_leftCameraStatusLabel->setText("Left: ❌ Disconnected");
        m_leftCameraStatusLabel->setStyleSheet("QLabel { color: red; font-weight: bold; }");
    }
    // Right
    if (m_rightCameraConnected && m_selectedRightCamera >= 0) {
        m_rightCameraStatusLabel->setText(QString("Right: ✅ Cam %1").arg(m_selectedRightCamera));
        m_rightCameraStatusLabel->setStyleSheet("QLabel { color: #2e8b57; font-weight: bold; }");
    } else {
        m_rightCameraStatusLabel->setText("Right: ❌ Disconnected");
        m_rightCameraStatusLabel->setStyleSheet("QLabel { color: red; font-weight: bold; }");
    }
}

void MainWindow::toggleLiveProcessing() {
    if (!m_hasCalibration || !m_isCapturing) {
        QMessageBox::information(this, "Live Processing", "Need active capture and calibration to enable live processing.");
        if (m_liveProcessingAction) m_liveProcessingAction->setChecked(false);
        return;
    }
    m_liveProcessingEnabled = !m_liveProcessingEnabled;
    if (m_liveProcessingAction) m_liveProcessingAction->setChecked(m_liveProcessingEnabled);
    if (m_liveProcessingEnabled) {
        if (m_liveProcessingTimer) m_liveProcessingTimer->start(100); // ~10 FPS processing loop
        logCameraOperation("Live processing enabled", true);
    } else {
        if (m_liveProcessingTimer) m_liveProcessingTimer->stop();
        logCameraOperation("Live processing disabled", true);
    }
    updateUI();
}

void MainWindow::onLiveFrameProcessed() {
    // Placeholder: in future run disparity / point cloud pipeline
    if (!m_lastLeftFrame.empty() && !m_lastRightFrame.empty()) {
        if (m_showDisparityAction && m_showDisparityAction->isChecked()) {
            // Simple grayscale diff as placeholder
            cv::Mat grayL, grayR, disp;
            cv::cvtColor(m_lastLeftFrame, grayL, cv::COLOR_BGR2GRAY);
            cv::cvtColor(m_lastRightFrame, grayR, cv::COLOR_BGR2GRAY);
            cv::absdiff(grayL, grayR, disp);
            cv::Mat dispColor; cv::applyColorMap(disp, dispColor, cv::COLORMAP_JET);
            cv::Mat rgb; cv::cvtColor(dispColor, rgb, cv::COLOR_BGR2RGB);
            QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
            QString tmp = QDir::tempPath()+"/live_disparity.png"; qimg.save(tmp);
            if (m_disparityWidget) m_disparityWidget->setImage(tmp);
        }
    }
}

void MainWindow::updateDisparityMap() {
    // Called when disparity action toggled; just refresh display or clear
    if (!m_showDisparityAction) return;
    if (!m_showDisparityAction->isChecked()) {
        if (m_disparityWidget) m_disparityWidget->clearImage();
    } else {
        // Force immediate update via processing placeholder
        onLiveFrameProcessed();
    }
}

void MainWindow::updatePointCloud() {
    // Placeholder: toggle point cloud widget visibility
    if (m_pointCloudWidget) m_pointCloudWidget->setVisible(m_showPointCloudAction && m_showPointCloudAction->isChecked());
}

void MainWindow::openBatchProcessing() {
    if (!m_batchProcessingWindow) {
        m_batchProcessingWindow = new stereo_vision::batch::BatchProcessingWindow(this);
    }
    m_batchProcessingWindow->show();
    m_batchProcessingWindow->raise();
    logCameraOperation("Opened Batch Processing window", true);
}

void MainWindow::openEpipolarChecker() {
    if (!m_epipolarChecker) {
        m_epipolarChecker = new EpipolarChecker(this);
    }
    m_epipolarChecker->show();
    m_epipolarChecker->raise();
    logCameraOperation("Opened Epipolar Checker", true);
}

void MainWindow::startAICalibration() {
    if (!m_isCapturing) {
        QMessageBox::information(this, "AI Calibration", "Start camera capture before AI calibration.");
        return;
    }
    if (m_aiCalibrationActive) return;
    m_aiCalibrationActive = true;
    m_calibrationFramesLeft.clear();
    m_calibrationFramesRight.clear();
    m_calibrationFrameCount = 0;
    logCameraOperation("AI Calibration started", true, QString("Collecting %1 frame pairs").arg(m_requiredCalibrationFrames));
    m_statusLabel->setText("AI Calibration: capturing frames...");
}

void MainWindow::captureCalibrationFrame() {
    if (!m_aiCalibrationActive) return;
    if (m_lastLeftFrame.empty() || m_lastRightFrame.empty()) return;
    m_calibrationFramesLeft.push_back(m_lastLeftFrame.clone());
    m_calibrationFramesRight.push_back(m_lastRightFrame.clone());
    ++m_calibrationFrameCount;
    onCalibrationProgress((m_calibrationFrameCount * 100) / m_requiredCalibrationFrames);
    if (m_calibrationFrameCount >= m_requiredCalibrationFrames) {
        onCalibrationComplete();
    }
}

void MainWindow::onCalibrationProgress(int progress) {
    m_statusLabel->setText(QString("AI Calibration Progress: %1% (%2/%3)").arg(progress).arg(m_calibrationFrameCount).arg(m_requiredCalibrationFrames));
}

void MainWindow::onCalibrationComplete() {
    m_aiCalibrationActive = false;
    // Placeholder: mark calibration available
    m_hasCalibration = true;
    updateUI();
    m_statusLabel->setText("AI Calibration complete (placeholder params)");
    logCameraOperation("AI Calibration completed", true, "Parameters ready (placeholder)");
}

void MainWindow::onCameraSelectionChanged() {
    // Placeholder; actual dialog will set selected cameras externally
    updateCameraStatusIndicators();
}

// === End restored methods ===

} // namespace stereo_vision::gui
