#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWidget>

namespace stereo_vision::gui {

struct StereoParameters {
  // SGBM parameters
  int minDisparity = 0;
  int numDisparities = 64;
  int blockSize = 9;
  int P1 = 8;
  int P2 = 32;
  int disp12MaxDiff = 1;
  int preFilterCap = 63;
  int uniquenessRatio = 10;
  int speckleWindowSize = 100;
  int speckleRange = 32;
  int mode = 0; // SGBM mode

  // Post-processing
  bool enableSpeckleFilter = true;
  bool enableMedianFilter = true;
  int medianKernelSize = 5;

  // Point cloud generation
  double scaleFactor = 1.0;
  bool enableColorMapping = true;
  bool enableFiltering = true;
  double maxDepth = 10.0; // meters
  double minDepth = 0.1;  // meters
};

class ParameterPanel : public QWidget {
  Q_OBJECT

public:
  explicit ParameterPanel(QWidget *parent = nullptr);
  ~ParameterPanel();

  StereoParameters getParameters() const { return m_parameters; }
  void setParameters(const StereoParameters &params);

  void loadSettings();
  void saveSettings();
  void resetToDefaults();

signals:
  void parametersChanged();

private slots:
  void onParameterChanged();
  void onResetClicked();
  void onPresetChanged(const QString &preset);

private:
  void setupUI();
  void setupSGBMGroup();
  void setupPostProcessingGroup();
  void setupPointCloudGroup();
  void setupPresets();
  void connectSignals();
  void updateUI();
  void applyPreset(const QString &preset);

  // Parameter storage
  StereoParameters m_parameters;

  // UI Layout
  QVBoxLayout *m_mainLayout;

  // SGBM Parameters Group
  QGroupBox *m_sgbmGroup;

  // SGBM controls (labels handled by QFormLayout)
  QSpinBox *m_minDisparitySpin;
  QSpinBox *m_numDisparitiesSpin;
  QSpinBox *m_blockSizeSpin;
  QSpinBox *m_p1Spin;
  QSpinBox *m_p2Spin;
  QSpinBox *m_disp12MaxDiffSpin;
  QSpinBox *m_preFilterCapSpin;
  QSpinBox *m_uniquenessRatioSpin;
  QSpinBox *m_speckleWindowSizeSpin;
  QSpinBox *m_speckleRangeSpin;
  QComboBox *m_modeCombo;

  // Post-processing Group
  QGroupBox *m_postProcessGroup;
  QCheckBox *m_enableSpeckleFilterCheck;
  QCheckBox *m_enableMedianFilterCheck;
  QSpinBox *m_medianKernelSizeSpin;

  // Point Cloud Group
  QGroupBox *m_pointCloudGroup;
  QDoubleSpinBox *m_scaleFactorSpin;
  QDoubleSpinBox *m_maxDepthSpin;
  QDoubleSpinBox *m_minDepthSpin;
  QCheckBox *m_enableColorMappingCheck;
  QCheckBox *m_enableFilteringCheck;

  // Control buttons
  QHBoxLayout *m_buttonLayout;
  QComboBox *m_presetCombo;
  QPushButton *m_resetButton;

  // Settings
  QSettings *m_settings;
};

} // namespace stereo_vision::gui
