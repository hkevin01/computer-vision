#include "camera_manager.hpp"
#include "gui/image_display_widget.hpp"
#include "gui/live_calibration_widget.hpp"
#include <QComboBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

LiveCalibrationWidget::LiveCalibrationWidget(QWidget *parent)
    : QWidget(parent),
      m_camera_manager(std::make_unique<stereo_vision::CameraManager>()),
      m_live_view_timer(new QTimer(this)) {
  setupUI();
}

LiveCalibrationWidget::~LiveCalibrationWidget() = default;

void LiveCalibrationWidget::detectCameras() {
  // Clear existing camera list
  m_left_camera_selector->clear();
  m_right_camera_selector->clear();

  // Detect available cameras (simplified implementation)
  // In a real implementation, this would use platform-specific camera detection
  for (int i = 0; i < 4; ++i) {
    cv::VideoCapture cap(i);
    if (cap.isOpened()) {
      QString cameraName = QString("Camera %1").arg(i);
      m_left_camera_selector->addItem(cameraName, i);
      m_right_camera_selector->addItem(cameraName, i);
      cap.release();
    }
  }

  // Set different defaults if possible
  if (m_left_camera_selector->count() >= 2) {
    m_left_camera_selector->setCurrentIndex(0);
    m_right_camera_selector->setCurrentIndex(1);
  }

  // Enable capture button if we have at least one camera
  m_capture_button->setEnabled(m_left_camera_selector->count() > 0);

  if (m_left_camera_selector->count() == 0) {
    QMessageBox::warning(this, "No Cameras", "No cameras were detected.");
  }
}

void LiveCalibrationWidget::captureImages() {
  // Get selected camera IDs
  int leftCameraId = m_left_camera_selector->currentData().toInt();
  int rightCameraId = m_right_camera_selector->currentData().toInt();

  if (leftCameraId == rightCameraId) {
    QMessageBox::warning(this, "Error",
                         "Please select different cameras for left and right.");
    return;
  }

  // Capture frames
  cv::VideoCapture leftCap(leftCameraId);
  cv::VideoCapture rightCap(rightCameraId);

  if (!leftCap.isOpened() || !rightCap.isOpened()) {
    QMessageBox::warning(this, "Error", "Failed to open one or both cameras.");
    return;
  }

  cv::Mat leftFrame, rightFrame;
  leftCap >> leftFrame;
  rightCap >> rightFrame;

  if (!leftFrame.empty() && !rightFrame.empty()) {
    // Store the captured pair
    m_captured_pairs.emplace_back(leftFrame.clone(), rightFrame.clone());

    // Display the captured images
    m_left_camera_view->setImage(leftFrame);
    m_right_camera_view->setImage(rightFrame);

    // Update capture count
    m_capture_count_label->setText(
        QString("Captured: %1 pairs").arg(m_captured_pairs.size()));

    // Enable calibration button if we have enough pairs
    m_start_calibration_button->setEnabled(m_captured_pairs.size() >= 5);
  } else {
    QMessageBox::warning(this, "Error",
                         "Failed to capture from one or both cameras.");
  }

  leftCap.release();
  rightCap.release();
}

void LiveCalibrationWidget::startCalibration() {
  if (m_captured_pairs.size() < 5) {
    QMessageBox::warning(this, "Error",
                         "Need at least 5 image pairs for calibration.");
    return;
  }

  QMessageBox::information(
      this, "Calibration",
      QString("Starting calibration with %1 image pairs.\nThis is a "
              "placeholder - actual calibration implementation needed.")
          .arg(m_captured_pairs.size()));
}

void LiveCalibrationWidget::resetCalibration() {
  m_captured_pairs.clear();
  m_capture_count_label->setText("Captured: 0 pairs");
  m_start_calibration_button->setEnabled(false);

  // Clear image displays
  m_left_camera_view->clearImage();
  m_right_camera_view->clearImage();
}

void LiveCalibrationWidget::setupUI() {
  m_layout = new QGridLayout(this);

  // Camera detection
  m_detect_cameras_button = new QPushButton("Detect Cameras", this);
  m_layout->addWidget(m_detect_cameras_button, 0, 0, 1, 2);

  // Camera selection
  m_layout->addWidget(new QLabel("Left Camera:", this), 1, 0);
  m_left_camera_selector = new QComboBox(this);
  m_layout->addWidget(m_left_camera_selector, 1, 1);

  m_layout->addWidget(new QLabel("Right Camera:", this), 2, 0);
  m_right_camera_selector = new QComboBox(this);
  m_layout->addWidget(m_right_camera_selector, 2, 1);

  // Camera views
  m_left_camera_view = new ImageDisplayWidget(this);
  m_left_camera_view->setMinimumSize(320, 240);
  m_layout->addWidget(m_left_camera_view, 3, 0);

  m_right_camera_view = new ImageDisplayWidget(this);
  m_right_camera_view->setMinimumSize(320, 240);
  m_layout->addWidget(m_right_camera_view, 3, 1);

  // Control buttons
  auto *buttonLayout = new QHBoxLayout();

  m_capture_button = new QPushButton("Capture Pair", this);
  m_capture_button->setEnabled(false);
  buttonLayout->addWidget(m_capture_button);

  m_start_calibration_button = new QPushButton("Start Calibration", this);
  m_start_calibration_button->setEnabled(false);
  buttonLayout->addWidget(m_start_calibration_button);

  m_reset_button = new QPushButton("Reset", this);
  buttonLayout->addWidget(m_reset_button);

  m_layout->addLayout(buttonLayout, 4, 0, 1, 2);

  // Status
  m_capture_count_label = new QLabel("Captured: 0 pairs", this);
  m_layout->addWidget(m_capture_count_label, 5, 0, 1, 2);

  // Connect signals
  connect(m_detect_cameras_button, &QPushButton::clicked, this,
          &LiveCalibrationWidget::detectCameras);
  connect(m_capture_button, &QPushButton::clicked, this,
          &LiveCalibrationWidget::captureImages);
  connect(m_start_calibration_button, &QPushButton::clicked, this,
          &LiveCalibrationWidget::startCalibration);
  connect(m_reset_button, &QPushButton::clicked, this,
          &LiveCalibrationWidget::resetCalibration);
}

} // namespace stereo_vision::gui
