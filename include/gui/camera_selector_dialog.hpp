#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <memory>

namespace stereo_vision {
class CameraManager;
}

namespace stereo_vision::gui {

class CameraSelectorDialog : public QDialog {
  Q_OBJECT

public:
  explicit CameraSelectorDialog(
      std::shared_ptr<stereo_vision::CameraManager> cameraManager,
      QWidget *parent = nullptr);
  ~CameraSelectorDialog();

  // Get selected camera indices
  int getSelectedLeftCamera() const;
  int getSelectedRightCamera() const;

  // Check if cameras are configured
  bool areCamerasConfigured() const;

private slots:
  void onLeftCameraChanged(int index);
  void onRightCameraChanged(int index);
  void onTestCameras();
  void onRefreshCameras();
  void onPreviewToggled(bool enabled);
  void updatePreview();

private:
  void setupUI();
  void refreshCameraList();
  void updateCameraInfo();
  void testCameraConnection(int cameraIndex, const QString &name);

  // UI Components
  QGroupBox *m_cameraSelectionGroup;
  QComboBox *m_leftCameraCombo;
  QComboBox *m_rightCameraCombo;
  QLabel *m_leftCameraLabel;
  QLabel *m_rightCameraLabel;
  QLabel *m_leftStatusLabel;
  QLabel *m_rightStatusLabel;

  QGroupBox *m_previewGroup;
  QCheckBox *m_previewCheckBox;
  QLabel *m_leftPreviewLabel;
  QLabel *m_rightPreviewLabel;
  QTimer *m_previewTimer;

  QPushButton *m_testButton;
  QPushButton *m_refreshButton;
  QPushButton *m_okButton;
  QPushButton *m_cancelButton;

  // Camera management
  std::shared_ptr<stereo_vision::CameraManager> m_cameraManager;
  int m_selectedLeftCamera;
  int m_selectedRightCamera;
  bool m_previewEnabled;
  bool m_camerasConfigured;
};

} // namespace stereo_vision::gui
