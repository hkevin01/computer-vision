#pragma once

#include <QWidget>
#include <opencv2/core/mat.hpp>
#include <vector>

// Forward declarations
class QComboBox;
class QPushButton;
class QGridLayout;
class QLabel;

namespace stereo_vision {
class CameraManager;
}

namespace stereo_vision::gui {

class ImageDisplayWidget;

class LiveCalibrationWidget : public QWidget {
  Q_OBJECT

public:
  explicit LiveCalibrationWidget(QWidget *parent = nullptr);
  ~LiveCalibrationWidget();

private slots:
  void detectCameras();
  void captureImages();
  void startCalibration();
  void resetCalibration();

private:
  void setupUI();
  void updateCameraLists(int selected_left = 0, int selected_right = 1);

  // UI Elements
  QGridLayout *m_layout;
  ImageDisplayWidget *m_left_camera_view;
  ImageDisplayWidget *m_right_camera_view;
  QPushButton *m_detect_cameras_button;
  QPushButton *m_capture_button;
  QPushButton *m_start_calibration_button;
  QPushButton *m_reset_button;
  QComboBox *m_left_camera_selector;
  QComboBox *m_right_camera_selector;
  QLabel *m_capture_count_label;

  // Backend
  std::unique_ptr<stereo_vision::CameraManager> m_camera_manager;
  std::vector<std::pair<cv::Mat, cv::Mat>> m_captured_pairs;
  QTimer *m_live_view_timer;
};

} // namespace stereo_vision::gui
