#include "gui/parameter_panel.hpp"
#include <QApplication>
#include <QVBoxLayout>
#include <QFormLayout>

namespace stereo_vision::gui {

ParameterPanel::ParameterPanel(QWidget *parent)
    : QWidget(parent), m_settings(new QSettings(this)) {
  setupUI();
  connectSignals();
  loadSettings();
  setMinimumWidth(480); // Increased to accommodate form layouts
  setMinimumHeight(700); // Increased height for all controls
}

ParameterPanel::~ParameterPanel() { saveSettings(); }

void ParameterPanel::setParameters(const StereoParameters &params) {
  m_parameters = params;
  updateUI();
  emit parametersChanged();
}

void ParameterPanel::loadSettings() {
  m_settings->beginGroup("StereoParameters");

  m_parameters.minDisparity = m_settings->value("minDisparity", 0).toInt();
  m_parameters.numDisparities = m_settings->value("numDisparities", 64).toInt();
  m_parameters.blockSize = m_settings->value("blockSize", 9).toInt();
  m_parameters.P1 = m_settings->value("P1", 8).toInt();
  m_parameters.P2 = m_settings->value("P2", 32).toInt();
  m_parameters.disp12MaxDiff = m_settings->value("disp12MaxDiff", 1).toInt();
  m_parameters.preFilterCap = m_settings->value("preFilterCap", 63).toInt();
  m_parameters.uniquenessRatio =
      m_settings->value("uniquenessRatio", 10).toInt();
  m_parameters.speckleWindowSize =
      m_settings->value("speckleWindowSize", 100).toInt();
  m_parameters.speckleRange = m_settings->value("speckleRange", 32).toInt();
  m_parameters.mode = m_settings->value("mode", 0).toInt();

  m_parameters.enableSpeckleFilter =
      m_settings->value("enableSpeckleFilter", true).toBool();
  m_parameters.enableMedianFilter =
      m_settings->value("enableMedianFilter", true).toBool();
  m_parameters.medianKernelSize =
      m_settings->value("medianKernelSize", 5).toInt();

  m_parameters.scaleFactor = m_settings->value("scaleFactor", 1.0).toDouble();
  m_parameters.enableColorMapping =
      m_settings->value("enableColorMapping", true).toBool();
  m_parameters.enableFiltering =
      m_settings->value("enableFiltering", true).toBool();
  m_parameters.maxDepth = m_settings->value("maxDepth", 10.0).toDouble();
  m_parameters.minDepth = m_settings->value("minDepth", 0.1).toDouble();

  m_settings->endGroup();
  updateUI();
}

void ParameterPanel::saveSettings() {
  m_settings->beginGroup("StereoParameters");

  m_settings->setValue("minDisparity", m_parameters.minDisparity);
  m_settings->setValue("numDisparities", m_parameters.numDisparities);
  m_settings->setValue("blockSize", m_parameters.blockSize);
  m_settings->setValue("P1", m_parameters.P1);
  m_settings->setValue("P2", m_parameters.P2);
  m_settings->setValue("disp12MaxDiff", m_parameters.disp12MaxDiff);
  m_settings->setValue("preFilterCap", m_parameters.preFilterCap);
  m_settings->setValue("uniquenessRatio", m_parameters.uniquenessRatio);
  m_settings->setValue("speckleWindowSize", m_parameters.speckleWindowSize);
  m_settings->setValue("speckleRange", m_parameters.speckleRange);
  m_settings->setValue("mode", m_parameters.mode);

  m_settings->setValue("enableSpeckleFilter", m_parameters.enableSpeckleFilter);
  m_settings->setValue("enableMedianFilter", m_parameters.enableMedianFilter);
  m_settings->setValue("medianKernelSize", m_parameters.medianKernelSize);

  m_settings->setValue("scaleFactor", m_parameters.scaleFactor);
  m_settings->setValue("enableColorMapping", m_parameters.enableColorMapping);
  m_settings->setValue("enableFiltering", m_parameters.enableFiltering);
  m_settings->setValue("maxDepth", m_parameters.maxDepth);
  m_settings->setValue("minDepth", m_parameters.minDepth);

  m_settings->endGroup();
}

void ParameterPanel::resetToDefaults() {
  m_parameters = StereoParameters(); // Reset to default values
  updateUI();
  emit parametersChanged();
}

void ParameterPanel::onParameterChanged() {
  // Update parameters from UI controls
  m_parameters.minDisparity = m_minDisparitySpin->value();
  m_parameters.numDisparities = m_numDisparitiesSpin->value();
  m_parameters.blockSize = m_blockSizeSpin->value();
  m_parameters.P1 = m_p1Spin->value();
  m_parameters.P2 = m_p2Spin->value();
  m_parameters.disp12MaxDiff = m_disp12MaxDiffSpin->value();
  m_parameters.preFilterCap = m_preFilterCapSpin->value();
  m_parameters.uniquenessRatio = m_uniquenessRatioSpin->value();
  m_parameters.speckleWindowSize = m_speckleWindowSizeSpin->value();
  m_parameters.speckleRange = m_speckleRangeSpin->value();
  m_parameters.mode = m_modeCombo->currentIndex();

  m_parameters.enableSpeckleFilter = m_enableSpeckleFilterCheck->isChecked();
  m_parameters.enableMedianFilter = m_enableMedianFilterCheck->isChecked();
  m_parameters.medianKernelSize = m_medianKernelSizeSpin->value();

  m_parameters.scaleFactor = m_scaleFactorSpin->value();
  m_parameters.enableColorMapping = m_enableColorMappingCheck->isChecked();
  m_parameters.enableFiltering = m_enableFilteringCheck->isChecked();
  m_parameters.maxDepth = m_maxDepthSpin->value();
  m_parameters.minDepth = m_minDepthSpin->value();

  emit parametersChanged();
}

void ParameterPanel::onResetClicked() { resetToDefaults(); }

void ParameterPanel::onPresetChanged(const QString &preset) {
  applyPreset(preset);
}

void ParameterPanel::setupUI() {
  m_mainLayout = new QVBoxLayout(this);

  setupSGBMGroup();
  setupPostProcessingGroup();
  setupPointCloudGroup();
  setupPresets();

  m_mainLayout->addWidget(m_sgbmGroup);
  m_mainLayout->addWidget(m_postProcessGroup);
  m_mainLayout->addWidget(m_pointCloudGroup);
  m_mainLayout->addLayout(m_buttonLayout);
  m_mainLayout->addStretch();
}

void ParameterPanel::setupSGBMGroup() {
  m_sgbmGroup = new QGroupBox("SGBM Parameters", this);

  // Use a form layout for better label/control pairing and prevent overlaps
  QFormLayout *formLayout = new QFormLayout(m_sgbmGroup);
  formLayout->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
  formLayout->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
  formLayout->setHorizontalSpacing(20);
  formLayout->setVerticalSpacing(10);
  formLayout->setContentsMargins(15, 15, 15, 15);
  formLayout->setFieldGrowthPolicy(QFormLayout::FieldsStayAtSizeHint);

  // Enhanced sizing for better readability - no overlap guaranteed
  m_sgbmGroup->setMinimumWidth(450);
  m_sgbmGroup->setMinimumHeight(400);
  m_sgbmGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

  auto configureSpin = [](QAbstractSpinBox *sb) {
    sb->setMinimumWidth(140);
    sb->setMaximumWidth(180);
    sb->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  };

  auto configureCombo = [](QComboBox *cb) {
    cb->setMinimumWidth(140);
    cb->setMaximumWidth(180);
    cb->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  };

  // Min Disparity
  m_minDisparitySpin = new QSpinBox(this);
  configureSpin(m_minDisparitySpin);
  m_minDisparitySpin->setRange(-128, 128);
  m_minDisparitySpin->setValue(m_parameters.minDisparity);
  formLayout->addRow("Min Disparity:", m_minDisparitySpin);

  // Num Disparities
  m_numDisparitiesSpin = new QSpinBox(this);
  configureSpin(m_numDisparitiesSpin);
  m_numDisparitiesSpin->setRange(16, 256);
  m_numDisparitiesSpin->setSingleStep(16);
  m_numDisparitiesSpin->setValue(m_parameters.numDisparities);
  formLayout->addRow("Num Disparities:", m_numDisparitiesSpin);

  // Block Size
  m_blockSizeSpin = new QSpinBox(this);
  configureSpin(m_blockSizeSpin);
  m_blockSizeSpin->setRange(3, 21);
  m_blockSizeSpin->setSingleStep(2);
  m_blockSizeSpin->setValue(m_parameters.blockSize);
  formLayout->addRow("Block Size:", m_blockSizeSpin);

  // P1
  m_p1Spin = new QSpinBox(this);
  configureSpin(m_p1Spin);
  m_p1Spin->setRange(0, 2000);
  m_p1Spin->setValue(m_parameters.P1);
  formLayout->addRow("P1:", m_p1Spin);

  // P2
  m_p2Spin = new QSpinBox(this);
  configureSpin(m_p2Spin);
  m_p2Spin->setRange(0, 4000);
  m_p2Spin->setValue(m_parameters.P2);
  formLayout->addRow("P2:", m_p2Spin);

  // Disp12 Max Diff
  m_disp12MaxDiffSpin = new QSpinBox(this);
  configureSpin(m_disp12MaxDiffSpin);
  m_disp12MaxDiffSpin->setRange(-1, 100);
  m_disp12MaxDiffSpin->setValue(m_parameters.disp12MaxDiff);
  formLayout->addRow("Disp12 Max Diff:", m_disp12MaxDiffSpin);

  // Pre Filter Cap
  m_preFilterCapSpin = new QSpinBox(this);
  configureSpin(m_preFilterCapSpin);
  m_preFilterCapSpin->setRange(1, 63);
  m_preFilterCapSpin->setValue(m_parameters.preFilterCap);
  formLayout->addRow("Pre Filter Cap:", m_preFilterCapSpin);

  // Uniqueness Ratio
  m_uniquenessRatioSpin = new QSpinBox(this);
  configureSpin(m_uniquenessRatioSpin);
  m_uniquenessRatioSpin->setRange(0, 100);
  m_uniquenessRatioSpin->setValue(m_parameters.uniquenessRatio);
  formLayout->addRow("Uniqueness Ratio:", m_uniquenessRatioSpin);

  // Speckle Window Size
  m_speckleWindowSizeSpin = new QSpinBox(this);
  configureSpin(m_speckleWindowSizeSpin);
  m_speckleWindowSizeSpin->setRange(0, 1000);
  m_speckleWindowSizeSpin->setValue(m_parameters.speckleWindowSize);
  formLayout->addRow("Speckle Window Size:", m_speckleWindowSizeSpin);

  // Speckle Range
  m_speckleRangeSpin = new QSpinBox(this);
  configureSpin(m_speckleRangeSpin);
  m_speckleRangeSpin->setRange(0, 100);
  m_speckleRangeSpin->setValue(m_parameters.speckleRange);
  formLayout->addRow("Speckle Range:", m_speckleRangeSpin);

  // Mode
  m_modeCombo = new QComboBox(this);
  configureCombo(m_modeCombo);
  m_modeCombo->addItems({"SGBM", "HH", "SGBM_3WAY", "HH4"});
  m_modeCombo->setCurrentIndex(m_parameters.mode);
  formLayout->addRow("Mode:", m_modeCombo);
}

void ParameterPanel::setupPostProcessingGroup() {
  m_postProcessGroup = new QGroupBox("Post Processing", this);
  m_postProcessGroup->setMinimumWidth(350);

  QFormLayout *formLayout = new QFormLayout(m_postProcessGroup);
  formLayout->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
  formLayout->setHorizontalSpacing(20);
  formLayout->setVerticalSpacing(8);
  formLayout->setContentsMargins(15, 10, 15, 10);

  m_enableSpeckleFilterCheck = new QCheckBox("Enable Speckle Filter", this);
  m_enableSpeckleFilterCheck->setChecked(m_parameters.enableSpeckleFilter);
  formLayout->addRow("", m_enableSpeckleFilterCheck);

  m_enableMedianFilterCheck = new QCheckBox("Enable Median Filter", this);
  m_enableMedianFilterCheck->setChecked(m_parameters.enableMedianFilter);
  formLayout->addRow("", m_enableMedianFilterCheck);

  m_medianKernelSizeSpin = new QSpinBox(this);
  m_medianKernelSizeSpin->setRange(3, 15);
  m_medianKernelSizeSpin->setSingleStep(2);
  m_medianKernelSizeSpin->setValue(m_parameters.medianKernelSize);
  m_medianKernelSizeSpin->setMinimumWidth(120);
  formLayout->addRow("Median Kernel Size:", m_medianKernelSizeSpin);
}

void ParameterPanel::setupPointCloudGroup() {
  m_pointCloudGroup = new QGroupBox("Point Cloud", this);
  m_pointCloudGroup->setMinimumWidth(350);

  QFormLayout *formLayout = new QFormLayout(m_pointCloudGroup);
  formLayout->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
  formLayout->setHorizontalSpacing(20);
  formLayout->setVerticalSpacing(8);
  formLayout->setContentsMargins(15, 10, 15, 10);

  // Scale Factor
  m_scaleFactorSpin = new QDoubleSpinBox(this);
  m_scaleFactorSpin->setRange(0.1, 10.0);
  m_scaleFactorSpin->setSingleStep(0.1);
  m_scaleFactorSpin->setDecimals(2);
  m_scaleFactorSpin->setValue(m_parameters.scaleFactor);
  m_scaleFactorSpin->setMinimumWidth(120);
  formLayout->addRow("Scale Factor:", m_scaleFactorSpin);

  // Max Depth
  m_maxDepthSpin = new QDoubleSpinBox(this);
  m_maxDepthSpin->setRange(0.1, 100.0);
  m_maxDepthSpin->setSingleStep(0.5);
  m_maxDepthSpin->setDecimals(1);
  m_maxDepthSpin->setValue(m_parameters.maxDepth);
  m_maxDepthSpin->setMinimumWidth(120);
  formLayout->addRow("Max Depth (m):", m_maxDepthSpin);

  // Min Depth
  m_minDepthSpin = new QDoubleSpinBox(this);
  m_minDepthSpin->setRange(0.01, 10.0);
  m_minDepthSpin->setSingleStep(0.1);
  m_minDepthSpin->setDecimals(2);
  m_minDepthSpin->setValue(m_parameters.minDepth);
  m_minDepthSpin->setMinimumWidth(120);
  formLayout->addRow("Min Depth (m):", m_minDepthSpin);

  m_enableColorMappingCheck = new QCheckBox("Enable Color Mapping", this);
  m_enableColorMappingCheck->setChecked(m_parameters.enableColorMapping);
  formLayout->addRow("", m_enableColorMappingCheck);

  m_enableFilteringCheck = new QCheckBox("Enable Filtering", this);
  m_enableFilteringCheck->setChecked(m_parameters.enableFiltering);
  formLayout->addRow("", m_enableFilteringCheck);
}

void ParameterPanel::setupPresets() {
  m_buttonLayout = new QHBoxLayout();

  m_presetCombo = new QComboBox(this);
  m_presetCombo->addItems(
      {"Default", "High Quality", "Fast", "Outdoor", "Indoor"});

  m_resetButton = new QPushButton("Reset", this);

  m_buttonLayout->addWidget(new QLabel("Preset:", this));
  m_buttonLayout->addWidget(m_presetCombo);
  m_buttonLayout->addStretch();
  m_buttonLayout->addWidget(m_resetButton);
}

void ParameterPanel::connectSignals() {
  // Connect all spin boxes
  connect(m_minDisparitySpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &ParameterPanel::onParameterChanged);
  connect(m_numDisparitiesSpin, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ParameterPanel::onParameterChanged);
  connect(m_blockSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &ParameterPanel::onParameterChanged);
  connect(m_p1Spin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &ParameterPanel::onParameterChanged);
  connect(m_p2Spin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &ParameterPanel::onParameterChanged);
  connect(m_disp12MaxDiffSpin, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ParameterPanel::onParameterChanged);
  connect(m_preFilterCapSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &ParameterPanel::onParameterChanged);
  connect(m_uniquenessRatioSpin, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ParameterPanel::onParameterChanged);
  connect(m_speckleWindowSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ParameterPanel::onParameterChanged);
  connect(m_speckleRangeSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &ParameterPanel::onParameterChanged);
  connect(m_medianKernelSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &ParameterPanel::onParameterChanged);

  // Connect double spin boxes
  connect(m_scaleFactorSpin,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          &ParameterPanel::onParameterChanged);
  connect(m_maxDepthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, &ParameterPanel::onParameterChanged);
  connect(m_minDepthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, &ParameterPanel::onParameterChanged);

  // Connect combo boxes
  connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &ParameterPanel::onParameterChanged);
  connect(m_presetCombo, &QComboBox::currentTextChanged, this,
          &ParameterPanel::onPresetChanged);

  // Connect check boxes
  connect(m_enableSpeckleFilterCheck, &QCheckBox::toggled, this,
          &ParameterPanel::onParameterChanged);
  connect(m_enableMedianFilterCheck, &QCheckBox::toggled, this,
          &ParameterPanel::onParameterChanged);
  connect(m_enableColorMappingCheck, &QCheckBox::toggled, this,
          &ParameterPanel::onParameterChanged);
  connect(m_enableFilteringCheck, &QCheckBox::toggled, this,
          &ParameterPanel::onParameterChanged);

  // Connect buttons
  connect(m_resetButton, &QPushButton::clicked, this,
          &ParameterPanel::onResetClicked);
}

void ParameterPanel::updateUI() {
  m_minDisparitySpin->setValue(m_parameters.minDisparity);
  m_numDisparitiesSpin->setValue(m_parameters.numDisparities);
  m_blockSizeSpin->setValue(m_parameters.blockSize);
  m_p1Spin->setValue(m_parameters.P1);
  m_p2Spin->setValue(m_parameters.P2);
  m_disp12MaxDiffSpin->setValue(m_parameters.disp12MaxDiff);
  m_preFilterCapSpin->setValue(m_parameters.preFilterCap);
  m_uniquenessRatioSpin->setValue(m_parameters.uniquenessRatio);
  m_speckleWindowSizeSpin->setValue(m_parameters.speckleWindowSize);
  m_speckleRangeSpin->setValue(m_parameters.speckleRange);
  m_modeCombo->setCurrentIndex(m_parameters.mode);

  m_enableSpeckleFilterCheck->setChecked(m_parameters.enableSpeckleFilter);
  m_enableMedianFilterCheck->setChecked(m_parameters.enableMedianFilter);
  m_medianKernelSizeSpin->setValue(m_parameters.medianKernelSize);

  m_scaleFactorSpin->setValue(m_parameters.scaleFactor);
  m_enableColorMappingCheck->setChecked(m_parameters.enableColorMapping);
  m_enableFilteringCheck->setChecked(m_parameters.enableFiltering);
  m_maxDepthSpin->setValue(m_parameters.maxDepth);
  m_minDepthSpin->setValue(m_parameters.minDepth);
}

void ParameterPanel::applyPreset(const QString &preset) {
  if (preset == "High Quality") {
    m_parameters.numDisparities = 128;
    m_parameters.blockSize = 11;
    m_parameters.P1 = 24;
    m_parameters.P2 = 96;
    m_parameters.uniquenessRatio = 15;
    m_parameters.speckleWindowSize = 150;
    m_parameters.enableSpeckleFilter = true;
    m_parameters.enableMedianFilter = true;
  } else if (preset == "Fast") {
    m_parameters.numDisparities = 32;
    m_parameters.blockSize = 5;
    m_parameters.P1 = 8;
    m_parameters.P2 = 32;
    m_parameters.uniquenessRatio = 5;
    m_parameters.speckleWindowSize = 50;
    m_parameters.enableSpeckleFilter = false;
    m_parameters.enableMedianFilter = false;
  } else if (preset == "Outdoor") {
    m_parameters.numDisparities = 96;
    m_parameters.blockSize = 9;
    m_parameters.P1 = 16;
    m_parameters.P2 = 64;
    m_parameters.maxDepth = 50.0;
    m_parameters.minDepth = 1.0;
  } else if (preset == "Indoor") {
    m_parameters.numDisparities = 64;
    m_parameters.blockSize = 7;
    m_parameters.P1 = 12;
    m_parameters.P2 = 48;
    m_parameters.maxDepth = 10.0;
    m_parameters.minDepth = 0.3;
  } else {
    // Default preset
    m_parameters = StereoParameters();
  }

  updateUI();
  emit parametersChanged();
}

} // namespace stereo_vision::gui
