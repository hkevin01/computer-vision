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
  QGridLayout *m_sgbmLayout;

  QLabel *m_minDisparityLabel;
  QSpinBox *m_minDisparitySpin;
  QSlider *m_minDisparitySlider;

  QLabel *m_numDisparitiesLabel;
  QSpinBox *m_numDisparitiesSpin;
  QSlider *m_numDisparitiesSlider;

  QLabel *m_blockSizeLabel;
  QSpinBox *m_blockSizeSpin;
  QSlider *m_blockSizeSlider;

  QLabel *m_p1Label;
  QSpinBox *m_p1Spin;
  QSlider *m_p1Slider;

  QLabel *m_p2Label;
  QSpinBox *m_p2Spin;
  QSlider *m_p2Slider;

  QLabel *m_disp12MaxDiffLabel;
  QSpinBox *m_disp12MaxDiffSpin;
  QSlider *m_disp12MaxDiffSlider;

  QLabel *m_preFilterCapLabel;
  QSpinBox *m_preFilterCapSpin;
  QSlider *m_preFilterCapSlider;

  QLabel *m_uniquenessRatioLabel;
  QSpinBox *m_uniquenessRatioSpin;
  QSlider *m_uniquenessRatioSlider;

  QLabel *m_speckleWindowSizeLabel;
  QSpinBox *m_speckleWindowSizeSpin;
  QSlider *m_speckleWindowSizeSlider;

  QLabel *m_speckleRangeLabel;
  QSpinBox *m_speckleRangeSpin;
  QSlider *m_speckleRangeSlider;

  QLabel *m_modeLabel;
  QComboBox *m_modeCombo;

  // Post-processing Group
  QGroupBox *m_postProcessGroup;
  QVBoxLayout *m_postProcessLayout;

  QCheckBox *m_enableSpeckleFilterCheck;
  QCheckBox *m_enableMedianFilterCheck;

  QLabel *m_medianKernelSizeLabel;
  QSpinBox *m_medianKernelSizeSpin;
  QSlider *m_medianKernelSizeSlider;

  // Point Cloud Group
  QGroupBox *m_pointCloudGroup;
  QGridLayout *m_pointCloudLayout;

  QLabel *m_scaleFactorLabel;
  QDoubleSpinBox *m_scaleFactorSpin;

  QCheckBox *m_enableColorMappingCheck;
  QCheckBox *m_enableFilteringCheck;

  QLabel *m_maxDepthLabel;
  QDoubleSpinBox *m_maxDepthSpin;

  QLabel *m_minDepthLabel;
  QDoubleSpinBox *m_minDepthSpin;

  // Control buttons
  QHBoxLayout *m_buttonLayout;
  QComboBox *m_presetCombo;
  QPushButton *m_resetButton;

  // Settings
  QSettings *m_settings;
};

} // namespace stereo_vision::gui
