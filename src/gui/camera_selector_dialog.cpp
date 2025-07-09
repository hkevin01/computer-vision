#include "gui/camera_selector_dialog.hpp"
#include "camera_manager.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

CameraSelectorDialog::CameraSelectorDialog(
    std::shared_ptr<stereo_vision::CameraManager> cameraManager,
    QWidget *parent)
    : QDialog(parent), m_cameraManager(cameraManager), m_selectedLeftCamera(-1),
      m_selectedRightCamera(-1), m_previewEnabled(false),
      m_camerasConfigured(false), m_previewTimer(new QTimer(this)) {

  setWindowTitle("Camera Selection");
  setModal(true);
  resize(600, 500);

  setupUI();
  refreshCameraList();

  // Setup preview timer
  m_previewTimer->setInterval(100); // 10 FPS preview
  connect(m_previewTimer, &QTimer::timeout, this,
          &CameraSelectorDialog::updatePreview);
}

CameraSelectorDialog::~CameraSelectorDialog() {
  if (m_previewTimer->isActive()) {
    m_previewTimer->stop();
  }
}

void CameraSelectorDialog::setupUI() {
  auto *mainLayout = new QVBoxLayout(this);

  // Camera Selection Group
  m_cameraSelectionGroup = new QGroupBox("Camera Selection", this);
  auto *selectionLayout = new QGridLayout(m_cameraSelectionGroup);

  // Left Camera
  selectionLayout->addWidget(new QLabel("Left Camera:", this), 0, 0);
  m_leftCameraCombo = new QComboBox(this);
  m_leftCameraCombo->setMinimumWidth(200);
  selectionLayout->addWidget(m_leftCameraCombo, 0, 1);
  m_leftStatusLabel = new QLabel("Not connected", this);
  m_leftStatusLabel->setStyleSheet("color: red;");
  selectionLayout->addWidget(m_leftStatusLabel, 0, 2);

  // Right Camera
  selectionLayout->addWidget(new QLabel("Right Camera:", this), 1, 0);
  m_rightCameraCombo = new QComboBox(this);
  m_rightCameraCombo->setMinimumWidth(200);
  selectionLayout->addWidget(m_rightCameraCombo, 1, 1);
  m_rightStatusLabel = new QLabel("Not connected", this);
  m_rightStatusLabel->setStyleSheet("color: red;");
  selectionLayout->addWidget(m_rightStatusLabel, 1, 2);

  // Control buttons
  auto *buttonLayout = new QHBoxLayout();
  m_testButton = new QPushButton("Test Cameras", this);
  m_refreshButton = new QPushButton("Refresh", this);
  buttonLayout->addWidget(m_testButton);
  buttonLayout->addWidget(m_refreshButton);
  buttonLayout->addStretch();

  selectionLayout->addLayout(buttonLayout, 2, 0, 1, 3);
  mainLayout->addWidget(m_cameraSelectionGroup);

  // Preview Group
  m_previewGroup = new QGroupBox("Camera Preview", this);
  auto *previewLayout = new QVBoxLayout(m_previewGroup);

  m_previewCheckBox = new QCheckBox("Enable Preview", this);
  previewLayout->addWidget(m_previewCheckBox);

  auto *previewImageLayout = new QHBoxLayout();

  // Left preview
  auto *leftPreviewLayout = new QVBoxLayout();
  leftPreviewLayout->addWidget(new QLabel("Left Camera", this));
  m_leftPreviewLabel = new QLabel(this);
  m_leftPreviewLabel->setMinimumSize(160, 120);
  m_leftPreviewLabel->setStyleSheet(
      "border: 1px solid gray; background-color: black;");
  m_leftPreviewLabel->setAlignment(Qt::AlignCenter);
  m_leftPreviewLabel->setText("No Preview");
  leftPreviewLayout->addWidget(m_leftPreviewLabel);
  previewImageLayout->addLayout(leftPreviewLayout);

  // Right preview
  auto *rightPreviewLayout = new QVBoxLayout();
  rightPreviewLayout->addWidget(new QLabel("Right Camera", this));
  m_rightPreviewLabel = new QLabel(this);
  m_rightPreviewLabel->setMinimumSize(160, 120);
  m_rightPreviewLabel->setStyleSheet(
      "border: 1px solid gray; background-color: black;");
  m_rightPreviewLabel->setAlignment(Qt::AlignCenter);
  m_rightPreviewLabel->setText("No Preview");
  rightPreviewLayout->addWidget(m_rightPreviewLabel);
  previewImageLayout->addLayout(rightPreviewLayout);

  previewLayout->addLayout(previewImageLayout);
  mainLayout->addWidget(m_previewGroup);

  // Dialog buttons
  auto *dialogButtonLayout = new QHBoxLayout();
  m_okButton = new QPushButton("OK", this);
  m_cancelButton = new QPushButton("Cancel", this);
  m_okButton->setDefault(true);
  m_okButton->setEnabled(false); // Disabled until cameras are configured

  dialogButtonLayout->addStretch();
  dialogButtonLayout->addWidget(m_okButton);
  dialogButtonLayout->addWidget(m_cancelButton);
  mainLayout->addLayout(dialogButtonLayout);

  // Connect signals
  connect(m_leftCameraCombo,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &CameraSelectorDialog::onLeftCameraChanged);
  connect(m_rightCameraCombo,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &CameraSelectorDialog::onRightCameraChanged);
  connect(m_testButton, &QPushButton::clicked, this,
          &CameraSelectorDialog::onTestCameras);
  connect(m_refreshButton, &QPushButton::clicked, this,
          &CameraSelectorDialog::onRefreshCameras);
  connect(m_previewCheckBox, &QCheckBox::toggled, this,
          &CameraSelectorDialog::onPreviewToggled);
  connect(m_okButton, &QPushButton::clicked, this, &QDialog::accept);
  connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
}

void CameraSelectorDialog::refreshCameraList() {
  // Clear existing items
  m_leftCameraCombo->clear();
  m_rightCameraCombo->clear();

  if (!m_cameraManager) {
    QMessageBox::warning(this, "Error", "Camera manager not available");
    return;
  }

  // Detect available cameras
  int numCameras = m_cameraManager->detectCameras();
  const auto &cameraList = m_cameraManager->getCameraList();

  if (numCameras == 0) {
    m_leftCameraCombo->addItem("No cameras found");
    m_rightCameraCombo->addItem("No cameras found");
    QMessageBox::information(this, "No Cameras",
                             "No usable cameras detected.\n\n"
                             "Please check:\n"
                             "• Camera connections\n"
                             "• Camera permissions\n"
                             "• Other applications using cameras");
    return;
  }

  // Add option for no selection
  m_leftCameraCombo->addItem("(None)", -1);
  m_rightCameraCombo->addItem("(None)", -1);

  // Add cameras to combo boxes
  for (int i = 0; i < numCameras; ++i) {
    QString cameraName;
    if (i < static_cast<int>(cameraList.size())) {
      cameraName = QString::fromStdString(cameraList[i]);
    } else {
      cameraName = QString("Camera %1").arg(i);
    }

    m_leftCameraCombo->addItem(cameraName, i);
    m_rightCameraCombo->addItem(cameraName, i);
  }

  // Handle single camera case
  if (numCameras == 1) {
    QMessageBox::information(
        this, "Single Camera Detected",
        QString(
            "Only one camera detected: %1\n\n"
            "For stereo vision you typically need two cameras.\n"
            "You can still use this camera for mono capture by selecting it "
            "for either left or right channel.")
            .arg(QString::fromStdString(cameraList[0])));

    // Auto-select the camera for left channel
    m_leftCameraCombo->setCurrentIndex(
        1); // Index 1 is the first camera (after "(None)")
  } else if (numCameras >= 2) {
    // Auto-select first two cameras
    m_leftCameraCombo->setCurrentIndex(1);  // First camera
    m_rightCameraCombo->setCurrentIndex(2); // Second camera
  }

  // Update camera info display
  updateCameraInfo();
}

void CameraSelectorDialog::onLeftCameraChanged(int index) {
  if (index >= 0) {
    m_selectedLeftCamera = m_leftCameraCombo->itemData(index).toInt();
    updateCameraInfo();
  }
}

void CameraSelectorDialog::onRightCameraChanged(int index) {
  if (index >= 0) {
    m_selectedRightCamera = m_rightCameraCombo->itemData(index).toInt();
    updateCameraInfo();
  }
}

void CameraSelectorDialog::onTestCameras() {
  if (m_selectedLeftCamera < 0 && m_selectedRightCamera < 0) {
    QMessageBox::warning(this, "Error", "Please select at least one camera");
    return;
  }

  bool singleCameraMode = (m_selectedLeftCamera == m_selectedRightCamera &&
                           m_selectedLeftCamera >= 0);

  // Test camera connections
  bool success = false;

  if (singleCameraMode) {
    // Single camera mode
    success = m_cameraManager->openSingleCamera(m_selectedLeftCamera);

    if (success) {
      m_leftStatusLabel->setText("Connected (Single)");
      m_leftStatusLabel->setStyleSheet("color: green;");
      m_rightStatusLabel->setText("Connected (Single)");
      m_rightStatusLabel->setStyleSheet("color: green;");
      m_camerasConfigured = true;
      m_okButton->setEnabled(true);

      QMessageBox::information(
          this, "Success",
          "Camera connected successfully!\n\n"
          "Single camera mode: The same camera will be used "
          "for both left and right channels. This is useful "
          "for manual stereo capture.");
    } else {
      m_leftStatusLabel->setText("Failed");
      m_leftStatusLabel->setStyleSheet("color: red;");
      m_rightStatusLabel->setText("Failed");
      m_rightStatusLabel->setStyleSheet("color: red;");
      m_camerasConfigured = false;
      m_okButton->setEnabled(false);

      QMessageBox::warning(this, "Error", "Failed to connect to the camera");
    }
  } else {
    // Dual camera mode
    if (m_selectedLeftCamera < 0 || m_selectedRightCamera < 0) {
      QMessageBox::warning(
          this, "Error",
          "Please select both left and right cameras for dual camera mode");
      return;
    }

    success = m_cameraManager->openCameras(m_selectedLeftCamera,
                                           m_selectedRightCamera);

    if (success) {
      m_leftStatusLabel->setText("Connected");
      m_leftStatusLabel->setStyleSheet("color: green;");
      m_rightStatusLabel->setText("Connected");
      m_rightStatusLabel->setStyleSheet("color: green;");
      m_camerasConfigured = true;
      m_okButton->setEnabled(true);

      QMessageBox::information(this, "Success",
                               "Both cameras connected successfully!");
    } else {
      m_leftStatusLabel->setText("Failed");
      m_leftStatusLabel->setStyleSheet("color: red;");
      m_rightStatusLabel->setText("Failed");
      m_rightStatusLabel->setStyleSheet("color: red;");
      m_camerasConfigured = false;
      m_okButton->setEnabled(false);

      QMessageBox::warning(this, "Error",
                           "Failed to connect to one or both cameras");
    }
  }
}

void CameraSelectorDialog::onRefreshCameras() {
  // Stop preview if running
  if (m_previewTimer->isActive()) {
    m_previewTimer->stop();
    m_previewCheckBox->setChecked(false);
  }

  // Refresh camera list
  refreshCameraList();

  // Reset status
  m_leftStatusLabel->setText("Not connected");
  m_leftStatusLabel->setStyleSheet("color: red;");
  m_rightStatusLabel->setText("Not connected");
  m_rightStatusLabel->setStyleSheet("color: red;");
  m_camerasConfigured = false;
  m_okButton->setEnabled(false);
}

void CameraSelectorDialog::onPreviewToggled(bool enabled) {
  m_previewEnabled = enabled;

  if (enabled) {
    if ((m_selectedLeftCamera >= 0 || m_selectedRightCamera >= 0)) {
      bool singleCameraMode = (m_selectedLeftCamera == m_selectedRightCamera &&
                               m_selectedLeftCamera >= 0);

      bool success = false;
      if (singleCameraMode) {
        success = m_cameraManager->isAnyCameraOpen() ||
                  m_cameraManager->openSingleCamera(m_selectedLeftCamera);
      } else if (m_selectedLeftCamera >= 0 && m_selectedRightCamera >= 0) {
        success = m_cameraManager->areCamerasOpen() ||
                  m_cameraManager->openCameras(m_selectedLeftCamera,
                                               m_selectedRightCamera);
      }

      if (success) {
        m_previewTimer->start();
      } else {
        m_previewCheckBox->setChecked(false);
        QMessageBox::warning(this, "Error",
                             "Cannot start preview: cameras not connected");
      }
    } else {
      m_previewCheckBox->setChecked(false);
      QMessageBox::warning(this, "Error",
                           "Please select at least one camera first");
    }
  } else {
    m_previewTimer->stop();
    m_leftPreviewLabel->setText("No Preview");
    m_rightPreviewLabel->setText("No Preview");
  }
}

void CameraSelectorDialog::updatePreview() {
  if (!m_cameraManager || !m_cameraManager->isAnyCameraOpen()) {
    return;
  }

  bool singleCameraMode = (m_selectedLeftCamera == m_selectedRightCamera &&
                           m_selectedLeftCamera >= 0);

  if (singleCameraMode) {
    // Single camera mode - show same frame in both previews
    cv::Mat frame;
    if (m_cameraManager->grabSingleFrame(frame)) {
      if (!frame.empty()) {
        cv::Mat display;
        cv::resize(frame, display, cv::Size(160, 120));
        cv::cvtColor(display, display, cv::COLOR_BGR2RGB);

        QImage qImage(display.data, display.cols, display.rows, display.step,
                      QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(qImage);

        // Show same frame in both previews
        m_leftPreviewLabel->setPixmap(pixmap);
        m_rightPreviewLabel->setPixmap(pixmap);
      }
    }
  } else {
    // Dual camera mode
    cv::Mat leftFrame, rightFrame;
    if (m_cameraManager->grabFrames(leftFrame, rightFrame)) {
      // Convert and display left frame
      if (!leftFrame.empty()) {
        cv::Mat leftDisplay;
        cv::resize(leftFrame, leftDisplay, cv::Size(160, 120));
        cv::cvtColor(leftDisplay, leftDisplay, cv::COLOR_BGR2RGB);

        QImage leftQImage(leftDisplay.data, leftDisplay.cols, leftDisplay.rows,
                          leftDisplay.step, QImage::Format_RGB888);
        m_leftPreviewLabel->setPixmap(QPixmap::fromImage(leftQImage));
      }

      // Convert and display right frame
      if (!rightFrame.empty()) {
        cv::Mat rightDisplay;
        cv::resize(rightFrame, rightDisplay, cv::Size(160, 120));
        cv::cvtColor(rightDisplay, rightDisplay, cv::COLOR_BGR2RGB);

        QImage rightQImage(rightDisplay.data, rightDisplay.cols,
                           rightDisplay.rows, rightDisplay.step,
                           QImage::Format_RGB888);
        m_rightPreviewLabel->setPixmap(QPixmap::fromImage(rightQImage));
      }
    }
  }
}

void CameraSelectorDialog::updateCameraInfo() {
  // Check if different cameras are selected
  if (m_selectedLeftCamera >= 0 && m_selectedRightCamera >= 0) {
    if (m_selectedLeftCamera == m_selectedRightCamera) {
      QMessageBox::warning(
          this, "Warning",
          "Left and right cameras are the same. This may work for testing but "
          "is not recommended for stereo vision.");
    }
  }
}

int CameraSelectorDialog::getSelectedLeftCamera() const {
  return m_selectedLeftCamera;
}

int CameraSelectorDialog::getSelectedRightCamera() const {
  return m_selectedRightCamera;
}

bool CameraSelectorDialog::areCamerasConfigured() const {
  return m_camerasConfigured;
}

} // namespace stereo_vision::gui
