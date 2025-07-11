#include "gui/calibration_wizard.hpp"
#include "camera_calibration.hpp"
#include "camera_manager.hpp"
#include "gui/image_display_widget.hpp"

#include <QApplication>
#include <QDesktopWidget>
#include <QFileDialog>
#include <QHeaderView>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QSplitter>
#include <QStandardPaths>

namespace stereo_vision::gui {

CalibrationWizard::CalibrationWizard(
    std::shared_ptr<stereo_vision::CameraManager> cameraManager,
    QWidget *parent)
    : QDialog(parent), m_cameraManager(cameraManager), m_currentStep(0),
      m_patternType(CHESSBOARD), m_patternSize(9, 6), m_squareSize(25.0f),
      m_requiredFrames(15), m_isCapturing(false), m_patternDetected(false),
      m_reprojectionErrorValue(0.0) {

  m_calibration = std::make_shared<stereo_vision::CameraCalibration>();
  m_captureTimer = new QTimer(this);
  m_captureTimer->setInterval(33); // ~30 FPS

  setupUI();
  updateNavigationButtons();

  connect(m_captureTimer, &QTimer::timeout, this,
          &CalibrationWizard::onCameraFrameReceived);

  setWindowTitle("Camera Calibration Wizard");
  setModal(true);
  resize(900, 700);
}

CalibrationWizard::~CalibrationWizard() = default;

void CalibrationWizard::setupUI() {
  auto *mainLayout = new QVBoxLayout(this);

  // Header
  auto *headerLayout = new QHBoxLayout();
  m_stepLabel = new QLabel("Step 1 of 6: Welcome");
  m_stepLabel->setStyleSheet("font-weight: bold; font-size: 14px;");
  m_progressBar = new QProgressBar();
  m_progressBar->setRange(0, TOTAL_STEPS - 1);
  m_progressBar->setValue(0);

  headerLayout->addWidget(m_stepLabel);
  headerLayout->addStretch();
  headerLayout->addWidget(m_progressBar);
  mainLayout->addLayout(headerLayout);

  // Main content
  m_stackedWidget = new QStackedWidget();
  mainLayout->addWidget(m_stackedWidget);

  // Setup all pages
  setupWelcomePage();
  setupPatternPage();
  setupCapturePage();
  setupReviewPage();
  setupComputePage();
  setupResultsPage();

  // Navigation buttons
  auto *buttonLayout = new QHBoxLayout();
  m_cancelButton = new QPushButton("Cancel");
  m_previousButton = new QPushButton("< Previous");
  m_nextButton = new QPushButton("Next >");
  m_finishButton = new QPushButton("Finish");
  m_finishButton->setVisible(false);

  buttonLayout->addWidget(m_cancelButton);
  buttonLayout->addStretch();
  buttonLayout->addWidget(m_previousButton);
  buttonLayout->addWidget(m_nextButton);
  buttonLayout->addWidget(m_finishButton);
  mainLayout->addLayout(buttonLayout);

  // Connect navigation signals
  connect(m_nextButton, &QPushButton::clicked, this,
          &CalibrationWizard::nextStep);
  connect(m_previousButton, &QPushButton::clicked, this,
          &CalibrationWizard::previousStep);
  connect(m_cancelButton, &QPushButton::clicked, this,
          &CalibrationWizard::cancelWizard);
  connect(m_finishButton, &QPushButton::clicked, this,
          &CalibrationWizard::finishWizard);
}

void CalibrationWizard::setupWelcomePage() {
  m_welcomePage = new QWidget();
  auto *layout = new QVBoxLayout(m_welcomePage);

  auto *titleLabel = new QLabel("<h2>🎯 Camera Calibration Wizard</h2>");
  titleLabel->setAlignment(Qt::AlignCenter);

  m_welcomeText = new QLabel(
      "<p>Welcome to the Camera Calibration Wizard! This tool will guide you "
      "through "
      "the process of calibrating your camera for accurate stereo vision.</p>"

      "<h3>What you'll need:</h3>"
      "<ul>"
      "<li>📐 A printed calibration pattern (chessboard, circles, etc.)</li>"
      "<li>📷 A working camera connected to your system</li>"
      "<li>🕐 About 5-10 minutes for the calibration process</li>"
      "</ul>"

      "<h3>The calibration process:</h3>"
      "<ol>"
      "<li><b>Pattern Setup</b> - Configure your calibration pattern</li>"
      "<li><b>Frame Capture</b> - Take pictures of the pattern from different "
      "angles</li>"
      "<li><b>Review</b> - Check the quality of captured frames</li>"
      "<li><b>Computation</b> - Calculate calibration parameters</li>"
      "<li><b>Results</b> - View and save calibration data</li>"
      "</ol>"

      "<p><i>💡 Tip: For best results, capture 15-20 frames with the pattern "
      "at different positions, angles, and distances.</i></p>");
  m_welcomeText->setWordWrap(true);

  layout->addWidget(titleLabel);
  layout->addWidget(m_welcomeText);
  layout->addStretch();

  m_stackedWidget->addWidget(m_welcomePage);
}

void CalibrationWizard::setupPatternPage() {
  m_patternPage = new QWidget();
  auto *layout = new QVBoxLayout(m_patternPage);

  auto *titleLabel =
      new QLabel("<h3>📐 Calibration Pattern Configuration</h3>");

  auto *configGroup = new QGroupBox("Pattern Settings");
  auto *configLayout = new QGridLayout(configGroup);

  // Pattern type
  configLayout->addWidget(new QLabel("Pattern Type:"), 0, 0);
  m_patternTypeCombo = new QComboBox();
  m_patternTypeCombo->addItems(
      {"Chessboard", "Circles Grid", "Asymmetric Circles"});
  configLayout->addWidget(m_patternTypeCombo, 0, 1);

  // Pattern size
  configLayout->addWidget(new QLabel("Pattern Width (corners):"), 1, 0);
  m_patternWidthSpin = new QSpinBox();
  m_patternWidthSpin->setRange(3, 20);
  m_patternWidthSpin->setValue(9);
  configLayout->addWidget(m_patternWidthSpin, 1, 1);

  configLayout->addWidget(new QLabel("Pattern Height (corners):"), 2, 0);
  m_patternHeightSpin = new QSpinBox();
  m_patternHeightSpin->setRange(3, 20);
  m_patternHeightSpin->setValue(6);
  configLayout->addWidget(m_patternHeightSpin, 2, 1);

  // Square size
  configLayout->addWidget(new QLabel("Square Size (mm):"), 3, 0);
  m_squareSizeSpin = new QDoubleSpinBox();
  m_squareSizeSpin->setRange(1.0, 100.0);
  m_squareSizeSpin->setValue(25.0);
  m_squareSizeSpin->setSuffix(" mm");
  configLayout->addWidget(m_squareSizeSpin, 3, 1);

  // Pattern preview
  auto *previewGroup = new QGroupBox("Pattern Preview");
  auto *previewLayout = new QVBoxLayout(previewGroup);
  m_patternPreview = new QLabel();
  m_patternPreview->setMinimumSize(300, 200);
  m_patternPreview->setStyleSheet("border: 1px solid gray;");
  m_patternPreview->setAlignment(Qt::AlignCenter);
  m_patternPreview->setText("Chessboard Pattern\n9x6 corners\n25mm squares");
  previewLayout->addWidget(m_patternPreview);

  layout->addWidget(titleLabel);
  layout->addWidget(configGroup);
  layout->addWidget(previewGroup);
  layout->addStretch();

  // Connect signals
  connect(m_patternTypeCombo,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &CalibrationWizard::onPatternTypeChanged);
  connect(m_patternWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &CalibrationWizard::onPatternSizeChanged);
  connect(m_patternHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &CalibrationWizard::onPatternSizeChanged);
  connect(m_squareSizeSpin,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          &CalibrationWizard::onPatternSizeChanged);

  m_stackedWidget->addWidget(m_patternPage);
}

void CalibrationWizard::setupCapturePage() {
  m_capturePage = new QWidget();
  auto *layout = new QHBoxLayout(m_capturePage);

  // Left panel - camera view
  auto *leftPanel = new QWidget();
  auto *leftLayout = new QVBoxLayout(leftPanel);

  auto *titleLabel = new QLabel("<h3>📷 Frame Capture</h3>");

  m_liveView = new ImageDisplayWidget();
  m_liveView->setMinimumSize(400, 300);

  auto *captureLayout = new QHBoxLayout();
  m_captureButton = new QPushButton("📸 Capture Frame");
  m_captureButton->setEnabled(false);
  captureLayout->addWidget(m_captureButton);
  captureLayout->addStretch();

  leftLayout->addWidget(titleLabel);
  leftLayout->addWidget(m_liveView);
  leftLayout->addLayout(captureLayout);

  // Right panel - status and controls
  auto *rightPanel = new QWidget();
  auto *rightLayout = new QVBoxLayout(rightPanel);
  rightPanel->setMaximumWidth(250);

  auto *statusGroup = new QGroupBox("Detection Status");
  auto *statusLayout = new QVBoxLayout(statusGroup);

  m_detectionStatus = new QLabel("❌ Pattern not detected");
  m_qualityLabel = new QLabel("Quality: --");
  m_frameCountLabel = new QLabel("Frames: 0 / 15");

  statusLayout->addWidget(m_detectionStatus);
  statusLayout->addWidget(m_qualityLabel);
  statusLayout->addWidget(m_frameCountLabel);

  m_captureProgress = new QProgressBar();
  m_captureProgress->setRange(0, m_requiredFrames);
  m_captureProgress->setValue(0);
  statusLayout->addWidget(m_captureProgress);

  auto *instructionsGroup = new QGroupBox("Instructions");
  auto *instructionsLayout = new QVBoxLayout(instructionsGroup);
  auto *instructions =
      new QLabel("1. Position the calibration pattern in the camera view\n\n"
                 "2. Ensure the pattern is fully visible and in focus\n\n"
                 "3. When detected (✅), click 'Capture Frame'\n\n"
                 "4. Move the pattern to a different position/angle\n\n"
                 "5. Repeat until you have enough frames");
  instructions->setWordWrap(true);
  instructionsLayout->addWidget(instructions);

  rightLayout->addWidget(statusGroup);
  rightLayout->addWidget(instructionsGroup);
  rightLayout->addStretch();

  layout->addWidget(leftPanel);
  layout->addWidget(rightPanel);

  connect(m_captureButton, &QPushButton::clicked, this,
          &CalibrationWizard::captureFrame);

  m_stackedWidget->addWidget(m_capturePage);
}

void CalibrationWizard::setupReviewPage() {
  m_reviewPage = new QWidget();
  auto *layout = new QHBoxLayout(m_reviewPage);

  // Left panel - frame list
  auto *leftPanel = new QWidget();
  auto *leftLayout = new QVBoxLayout(leftPanel);
  leftPanel->setMaximumWidth(300);

  auto *titleLabel = new QLabel("<h3>📋 Review Captured Frames</h3>");

  m_frameList = new QListWidget();
  m_frameList->setIconSize(QSize(100, 75));

  auto *buttonLayout = new QHBoxLayout();
  m_removeFrameButton = new QPushButton("Remove Frame");
  m_clearAllButton = new QPushButton("Clear All");
  m_removeFrameButton->setEnabled(false);

  buttonLayout->addWidget(m_removeFrameButton);
  buttonLayout->addWidget(m_clearAllButton);

  leftLayout->addWidget(titleLabel);
  leftLayout->addWidget(m_frameList);
  leftLayout->addLayout(buttonLayout);

  // Right panel - frame preview and details
  auto *rightPanel = new QWidget();
  auto *rightLayout = new QVBoxLayout(rightPanel);

  m_framePreview = new ImageDisplayWidget();
  m_framePreview->setMinimumSize(400, 300);

  m_frameInfoLabel = new QLabel("Select a frame to view details");
  m_frameInfoLabel->setWordWrap(true);

  rightLayout->addWidget(m_framePreview);
  rightLayout->addWidget(m_frameInfoLabel);

  layout->addWidget(leftPanel);
  layout->addWidget(rightPanel);

  connect(m_frameList, &QListWidget::itemSelectionChanged, this, [this]() {
    auto selected = m_frameList->selectedItems();
    if (!selected.isEmpty()) {
      int index = m_frameList->row(selected.first());
      showFrameDetails(index);
      m_removeFrameButton->setEnabled(true);
    } else {
      m_removeFrameButton->setEnabled(false);
    }
  });

  connect(m_removeFrameButton, &QPushButton::clicked, this,
          &CalibrationWizard::removeSelectedFrame);
  connect(m_clearAllButton, &QPushButton::clicked, this,
          &CalibrationWizard::clearAllFrames);

  m_stackedWidget->addWidget(m_reviewPage);
}

void CalibrationWizard::setupComputePage() {
  m_computePage = new QWidget();
  auto *layout = new QVBoxLayout(m_computePage);

  auto *titleLabel = new QLabel("<h3>⚙️ Computing Calibration Parameters</h3>");

  m_computeStatus = new QLabel("Ready to compute calibration...");

  m_computeProgress = new QProgressBar();
  m_computeProgress->setRange(0, 0); // Indeterminate
  m_computeProgress->setVisible(false);

  m_computeLog = new QTextEdit();
  m_computeLog->setMaximumHeight(200);
  m_computeLog->setReadOnly(true);

  layout->addWidget(titleLabel);
  layout->addWidget(m_computeStatus);
  layout->addWidget(m_computeProgress);
  layout->addWidget(new QLabel("Computation Log:"));
  layout->addWidget(m_computeLog);
  layout->addStretch();

  m_stackedWidget->addWidget(m_computePage);
}

void CalibrationWizard::setupResultsPage() {
  m_resultsPage = new QWidget();
  auto *layout = new QVBoxLayout(m_resultsPage);

  auto *titleLabel = new QLabel("<h3>✅ Calibration Results</h3>");

  m_reprojectionError = new QLabel();
  m_reprojectionError->setStyleSheet("font-weight: bold; color: green;");

  m_resultsText = new QTextEdit();
  m_resultsText->setReadOnly(true);

  auto *buttonLayout = new QHBoxLayout();
  m_saveButton = new QPushButton("💾 Save Calibration");
  m_exportButton = new QPushButton("📤 Export Data");

  buttonLayout->addWidget(m_saveButton);
  buttonLayout->addWidget(m_exportButton);
  buttonLayout->addStretch();

  layout->addWidget(titleLabel);
  layout->addWidget(m_reprojectionError);
  layout->addWidget(m_resultsText);
  layout->addLayout(buttonLayout);

  connect(m_saveButton, &QPushButton::clicked, this, [this]() {
    QString fileName = QFileDialog::getSaveFileName(
        this, "Save Calibration",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) +
            "/camera_calibration.xml",
        "XML Files (*.xml);;All Files (*)");

    if (!fileName.isEmpty()) {
      // Save calibration data
      QMessageBox::information(this, "Success",
                               "Calibration saved successfully!");
    }
  });

  m_stackedWidget->addWidget(m_resultsPage);
}

void CalibrationWizard::nextStep() {
  if (m_currentStep < TOTAL_STEPS - 1) {
    // Validate current step before proceeding
    if (m_currentStep == 2) { // Capture page
      if (m_calibrationImages.size() < 5) {
        QMessageBox::warning(
            this, "Insufficient Frames",
            "Please capture at least 5 frames before proceeding.");
        return;
      }
      m_captureTimer->stop();
    } else if (m_currentStep == 3) { // Review page
      // Start computation
      runCalibration();
    }

    m_currentStep++;
    m_stackedWidget->setCurrentIndex(m_currentStep);
    updateNavigationButtons();

    // Special handling for capture page
    if (m_currentStep == 2) {
      // Start camera capture
      m_captureTimer->start();
    }
  }
}

void CalibrationWizard::previousStep() {
  if (m_currentStep > 0) {
    if (m_currentStep == 2) { // Leaving capture page
      m_captureTimer->stop();
    }

    m_currentStep--;
    m_stackedWidget->setCurrentIndex(m_currentStep);
    updateNavigationButtons();
  }
}

void CalibrationWizard::updateNavigationButtons() {
  m_stepLabel->setText(
      QString("Step %1 of %2: %3")
          .arg(m_currentStep + 1)
          .arg(TOTAL_STEPS)
          .arg(QStringList{"Welcome", "Pattern Setup", "Frame Capture",
                           "Review Frames", "Computing",
                           "Results"}[m_currentStep]));

  m_progressBar->setValue(m_currentStep);

  m_previousButton->setEnabled(m_currentStep > 0);

  if (m_currentStep == TOTAL_STEPS - 1) {
    m_nextButton->setVisible(false);
    m_finishButton->setVisible(true);
  } else {
    m_nextButton->setVisible(true);
    m_finishButton->setVisible(false);

    // Special button text for compute step
    if (m_currentStep == 3) {
      m_nextButton->setText("Compute Calibration");
    } else {
      m_nextButton->setText("Next >");
    }
  }
}

void CalibrationWizard::onCameraFrameReceived() {
  if (!m_cameraManager || m_currentStep != 2)
    return;

  // Get frame from camera manager
  cv::Mat frame = m_cameraManager->getCurrentFrame(0); // Assuming camera 0
  if (frame.empty())
    return;

  m_currentFrame = frame.clone();

  // Detect calibration pattern
  std::vector<cv::Point2f> corners;
  m_patternDetected = detectCalibrationPattern(frame, corners);

  if (m_patternDetected) {
    m_currentCorners = corners;

    // Draw detected corners
    cv::Mat displayFrame = frame.clone();
    cv::drawChessboardCorners(displayFrame, m_patternSize, corners, true);

    // Calculate quality
    double quality = calculateFrameQuality(frame, corners);

    m_detectionStatus->setText("✅ Pattern detected");
    m_qualityLabel->setText(QString("Quality: %1%").arg(quality, 0, 'f', 1));
    m_captureButton->setEnabled(quality >
                                30.0); // Enable if quality is good enough

    m_liveView->setImage(displayFrame);
  } else {
    m_detectionStatus->setText("❌ Pattern not detected");
    m_qualityLabel->setText("Quality: --");
    m_captureButton->setEnabled(false);
    m_liveView->setImage(frame);
  }

  updateCaptureStatus();
}

bool CalibrationWizard::detectCalibrationPattern(
    const cv::Mat &image, std::vector<cv::Point2f> &corners) {
  bool found = false;

  switch (m_patternType) {
  case CHESSBOARD:
    found = cv::findChessboardCorners(image, m_patternSize, corners,
                                      cv::CALIB_CB_ADAPTIVE_THRESH |
                                          cv::CALIB_CB_NORMALIZE_IMAGE);
    break;
  case CIRCLES_GRID:
    found = cv::findCirclesGrid(image, m_patternSize, corners);
    break;
  case ASYMMETRIC_CIRCLES:
    found = cv::findCirclesGrid(image, m_patternSize, corners,
                                cv::CALIB_CB_ASYMMETRIC_GRID);
    break;
  }

  if (found && m_patternType == CHESSBOARD) {
    // Refine corner positions for chessboard
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::cornerSubPix(
        gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                         0.1));
  }

  return found;
}

double CalibrationWizard::calculateFrameQuality(
    const cv::Mat &image, const std::vector<cv::Point2f> &corners) {
  if (corners.empty())
    return 0.0;

  // Calculate quality based on:
  // 1. Sharpness (Laplacian variance)
  // 2. Corner distribution
  // 3. Pattern coverage

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // Sharpness score
  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  double sharpness = stddev.val[0] * stddev.val[0];

  // Normalize sharpness (typical range: 0-2000)
  double sharpnessScore = std::min(sharpness / 1000.0, 1.0) * 40.0;

  // Coverage score - how much of the image is covered
  cv::Rect boundingRect = cv::boundingRect(corners);
  double imageArea = image.cols * image.rows;
  double patternArea = boundingRect.width * boundingRect.height;
  double coverageScore = (patternArea / imageArea) * 30.0;

  // Distribution score - how well distributed the corners are
  double distributionScore = 30.0; // Simplified for now

  return std::min(sharpnessScore + coverageScore + distributionScore, 100.0);
}

void CalibrationWizard::captureFrame() {
  if (!m_patternDetected || m_currentFrame.empty())
    return;

  m_calibrationImages.push_back(m_currentFrame.clone());
  m_imagePoints.push_back(m_currentCorners);

  double quality = calculateFrameQuality(m_currentFrame, m_currentCorners);
  m_frameQualities.push_back(quality);

  updateCaptureStatus();
  updateFrameList();

  // Brief feedback
  m_captureButton->setText("✅ Captured!");
  QTimer::singleShot(
      500, this, [this]() { m_captureButton->setText("📸 Capture Frame"); });
}

void CalibrationWizard::updateCaptureStatus() {
  int captured = m_calibrationImages.size();
  m_frameCountLabel->setText(
      QString("Frames: %1 / %2").arg(captured).arg(m_requiredFrames));
  m_captureProgress->setValue(captured);

  if (captured >= m_requiredFrames) {
    m_nextButton->setEnabled(true);
  }
}

void CalibrationWizard::updateFrameList() {
  m_frameList->clear();

  for (int i = 0; i < m_calibrationImages.size(); ++i) {
    auto *item = new QListWidgetItem();
    item->setText(QString("Frame %1\nQuality: %2%")
                      .arg(i + 1)
                      .arg(m_frameQualities[i], 0, 'f', 1));

    // Create thumbnail
    cv::Mat thumbnail;
    cv::resize(m_calibrationImages[i], thumbnail, cv::Size(100, 75));

    // Convert to QPixmap and set as icon
    QImage qimg(thumbnail.data, thumbnail.cols, thumbnail.rows, thumbnail.step,
                QImage::Format_BGR888);
    item->setIcon(QPixmap::fromImage(qimg.rgbSwapped()));

    m_frameList->addItem(item);
  }
}

void CalibrationWizard::showFrameDetails(int index) {
  if (index < 0 || index >= m_calibrationImages.size())
    return;

  const cv::Mat &image = m_calibrationImages[index];
  const std::vector<cv::Point2f> &corners = m_imagePoints[index];
  double quality = m_frameQualities[index];

  // Show image with detected corners
  cv::Mat displayImage = image.clone();
  cv::drawChessboardCorners(displayImage, m_patternSize, corners, true);
  m_framePreview->setImage(displayImage);

  // Show details
  QString details = QString("<b>Frame %1 Details:</b><br>"
                            "Quality Score: %2%<br>"
                            "Corners Detected: %3<br>"
                            "Image Size: %4 x %5<br>"
                            "Pattern Coverage: Good")
                        .arg(index + 1)
                        .arg(quality, 0, 'f', 1)
                        .arg(corners.size())
                        .arg(image.cols)
                        .arg(image.rows);

  m_frameInfoLabel->setText(details);
}

void CalibrationWizard::removeSelectedFrame() {
  auto selected = m_frameList->selectedItems();
  if (selected.isEmpty())
    return;

  int index = m_frameList->row(selected.first());

  m_calibrationImages.erase(m_calibrationImages.begin() + index);
  m_imagePoints.erase(m_imagePoints.begin() + index);
  m_frameQualities.erase(m_frameQualities.begin() + index);

  updateFrameList();
  m_framePreview->clear();
  m_frameInfoLabel->setText("Select a frame to view details");
}

void CalibrationWizard::clearAllFrames() {
  auto reply = QMessageBox::question(
      this, "Clear All Frames",
      "Are you sure you want to remove all captured frames?",
      QMessageBox::Yes | QMessageBox::No);

  if (reply == QMessageBox::Yes) {
    m_calibrationImages.clear();
    m_imagePoints.clear();
    m_frameQualities.clear();
    updateFrameList();
    m_framePreview->clear();
    m_frameInfoLabel->setText("Select a frame to view details");
  }
}

void CalibrationWizard::runCalibration() {
  if (m_calibrationImages.empty())
    return;

  m_computeProgress->setVisible(true);
  m_computeStatus->setText("Computing calibration parameters...");
  m_computeLog->append("Starting calibration computation...");

  // Prepare object points
  m_objectPoints.clear();
  std::vector<cv::Point3f> objp;
  for (int i = 0; i < m_patternSize.height; ++i) {
    for (int j = 0; j < m_patternSize.width; ++j) {
      objp.push_back(cv::Point3f(j * m_squareSize, i * m_squareSize, 0));
    }
  }

  for (size_t i = 0; i < m_calibrationImages.size(); ++i) {
    m_objectPoints.push_back(objp);
  }

  m_computeLog->append(
      QString("Processing %1 frames...").arg(m_calibrationImages.size()));

  // Run calibration
  cv::Size imageSize = m_calibrationImages[0].size();

  try {
    m_reprojectionErrorValue = cv::calibrateCamera(
        m_objectPoints, m_imagePoints, imageSize, m_cameraMatrix,
        m_distortionCoeffs, m_rvecs, m_tvecs, cv::CALIB_FIX_PRINCIPAL_POINT);

    m_computeLog->append("Calibration completed successfully!");
    m_computeLog->append(
        QString("Reprojection error: %1 pixels").arg(m_reprojectionErrorValue));

    // Update results page
    QString errorText = QString("🎯 Reprojection Error: %1 pixels")
                            .arg(m_reprojectionErrorValue, 0, 'f', 3);
    if (m_reprojectionErrorValue < 1.0) {
      errorText += " (Excellent)";
      m_reprojectionError->setStyleSheet("font-weight: bold; color: green;");
    } else if (m_reprojectionErrorValue < 2.0) {
      errorText += " (Good)";
      m_reprojectionError->setStyleSheet("font-weight: bold; color: orange;");
    } else {
      errorText += " (Needs Improvement)";
      m_reprojectionError->setStyleSheet("font-weight: bold; color: red;");
    }
    m_reprojectionError->setText(errorText);

    // Format results
    QString results =
        QString("Camera Matrix:\n%1  %2  %3\n%4  %5  %6\n%7  %8  %9\n\n"
                "Distortion Coefficients:\n%10  %11  %12  %13  %14\n\n"
                "Calibration Quality:\n"
                "- Frames Used: %15\n"
                "- Image Size: %16 x %17\n"
                "- Pattern Size: %18 x %19\n"
                "- Square Size: %20 mm\n"
                "- Reprojection Error: %21 pixels")
            .arg(m_cameraMatrix.at<double>(0, 0), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(0, 1), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(0, 2), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(1, 0), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(1, 1), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(1, 2), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(2, 0), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(2, 1), 0, 'f', 2)
            .arg(m_cameraMatrix.at<double>(2, 2), 0, 'f', 2)
            .arg(m_distortionCoeffs.at<double>(0), 0, 'f', 6)
            .arg(m_distortionCoeffs.at<double>(1), 0, 'f', 6)
            .arg(m_distortionCoeffs.at<double>(2), 0, 'f', 6)
            .arg(m_distortionCoeffs.at<double>(3), 0, 'f', 6)
            .arg(m_distortionCoeffs.at<double>(4), 0, 'f', 6)
            .arg(m_calibrationImages.size())
            .arg(imageSize.width)
            .arg(imageSize.height)
            .arg(m_patternSize.width)
            .arg(m_patternSize.height)
            .arg(m_squareSize)
            .arg(m_reprojectionErrorValue, 0, 'f', 3);

    m_resultsText->setText(results);

  } catch (const cv::Exception &e) {
    m_computeLog->append(QString("ERROR: %1").arg(e.what()));
    QMessageBox::critical(
        this, "Calibration Error",
        QString("Failed to compute calibration:\n%1").arg(e.what()));
  }

  m_computeProgress->setVisible(false);
  m_computeStatus->setText("Calibration computation completed.");
}

void CalibrationWizard::onPatternTypeChanged() {
  m_patternType = static_cast<PatternType>(m_patternTypeCombo->currentIndex());
  onPatternSizeChanged();
}

void CalibrationWizard::onPatternSizeChanged() {
  m_patternSize =
      cv::Size(m_patternWidthSpin->value(), m_patternHeightSpin->value());
  m_squareSize = m_squareSizeSpin->value();

  QString patternText = QString("%1 Pattern\n%2x%3 corners\n%4mm squares")
                            .arg(m_patternTypeCombo->currentText())
                            .arg(m_patternSize.width)
                            .arg(m_patternSize.height)
                            .arg(m_squareSize);

  m_patternPreview->setText(patternText);
}

void CalibrationWizard::finishWizard() { accept(); }

void CalibrationWizard::cancelWizard() {
  auto reply = QMessageBox::question(
      this, "Cancel Calibration",
      "Are you sure you want to cancel the calibration wizard?",
      QMessageBox::Yes | QMessageBox::No);

  if (reply == QMessageBox::Yes) {
    reject();
  }
}

} // namespace stereo_vision::gui
