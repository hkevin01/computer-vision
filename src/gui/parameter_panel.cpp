#include "gui/parameter_panel.hpp"
#include <QtWidgets/QApplication>

ParameterPanel::ParameterPanel(QWidget *parent)
    : QWidget(parent)
    , m_settings(new QSettings(this))
{
    setupUI();
    connectSignals();
    loadSettings();
}

ParameterPanel::~ParameterPanel()
{
    saveSettings();
}

void ParameterPanel::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    
    setupSGBMGroup();
    setupPostProcessingGroup();
    setupPointCloudGroup();
    setupPresets();
    
    m_mainLayout->addStretch();
    setLayout(m_mainLayout);
}

void ParameterPanel::setupSGBMGroup()
{
    m_sgbmGroup = new QGroupBox("SGBM Parameters");
    m_sgbmLayout = new QGridLayout(m_sgbmGroup);
    
    int row = 0;
    
    // Min Disparity
    m_minDisparityLabel = new QLabel("Min Disparity:");
    m_minDisparitySpin = new QSpinBox();
    m_minDisparitySpin->setRange(-128, 128);
    m_minDisparitySpin->setValue(m_parameters.minDisparity);
    m_minDisparitySlider = new QSlider(Qt::Horizontal);
    m_minDisparitySlider->setRange(-128, 128);
    m_minDisparitySlider->setValue(m_parameters.minDisparity);
    
    m_sgbmLayout->addWidget(m_minDisparityLabel, row, 0);
    m_sgbmLayout->addWidget(m_minDisparitySpin, row, 1);
    m_sgbmLayout->addWidget(m_minDisparitySlider, row, 2);
    row++;
    
    // Num Disparities
    m_numDisparitiesLabel = new QLabel("Num Disparities:");
    m_numDisparitiesSpin = new QSpinBox();
    m_numDisparitiesSpin->setRange(16, 256);
    m_numDisparitiesSpin->setSingleStep(16);
    m_numDisparitiesSpin->setValue(m_parameters.numDisparities);
    m_numDisparitiesSlider = new QSlider(Qt::Horizontal);
    m_numDisparitiesSlider->setRange(16, 256);
    m_numDisparitiesSlider->setValue(m_parameters.numDisparities);
    
    m_sgbmLayout->addWidget(m_numDisparitiesLabel, row, 0);
    m_sgbmLayout->addWidget(m_numDisparitiesSpin, row, 1);
    m_sgbmLayout->addWidget(m_numDisparitiesSlider, row, 2);
    row++;
    
    // Block Size
    m_blockSizeLabel = new QLabel("Block Size:");
    m_blockSizeSpin = new QSpinBox();
    m_blockSizeSpin->setRange(5, 21);
    m_blockSizeSpin->setSingleStep(2);
    m_blockSizeSpin->setValue(m_parameters.blockSize);
    m_blockSizeSlider = new QSlider(Qt::Horizontal);
    m_blockSizeSlider->setRange(5, 21);
    m_blockSizeSlider->setValue(m_parameters.blockSize);
    
    m_sgbmLayout->addWidget(m_blockSizeLabel, row, 0);
    m_sgbmLayout->addWidget(m_blockSizeSpin, row, 1);
    m_sgbmLayout->addWidget(m_blockSizeSlider, row, 2);
    row++;
    
    // P1
    m_p1Label = new QLabel("P1:");
    m_p1Spin = new QSpinBox();
    m_p1Spin->setRange(0, 1000);
    m_p1Spin->setValue(m_parameters.P1);
    m_p1Slider = new QSlider(Qt::Horizontal);
    m_p1Slider->setRange(0, 1000);
    m_p1Slider->setValue(m_parameters.P1);
    
    m_sgbmLayout->addWidget(m_p1Label, row, 0);
    m_sgbmLayout->addWidget(m_p1Spin, row, 1);
    m_sgbmLayout->addWidget(m_p1Slider, row, 2);
    row++;
    
    // P2
    m_p2Label = new QLabel("P2:");
    m_p2Spin = new QSpinBox();
    m_p2Spin->setRange(0, 10000);
    m_p2Spin->setValue(m_parameters.P2);
    m_p2Slider = new QSlider(Qt::Horizontal);
    m_p2Slider->setRange(0, 1000);
    m_p2Slider->setValue(std::min(m_parameters.P2, 1000));
    
    m_sgbmLayout->addWidget(m_p2Label, row, 0);
    m_sgbmLayout->addWidget(m_p2Spin, row, 1);
    m_sgbmLayout->addWidget(m_p2Slider, row, 2);
    row++;
    
    // Uniqueness Ratio
    m_uniquenessRatioLabel = new QLabel("Uniqueness Ratio:");
    m_uniquenessRatioSpin = new QSpinBox();
    m_uniquenessRatioSpin->setRange(0, 100);
    m_uniquenessRatioSpin->setValue(m_parameters.uniquenessRatio);
    m_uniquenessRatioSlider = new QSlider(Qt::Horizontal);
    m_uniquenessRatioSlider->setRange(0, 100);
    m_uniquenessRatioSlider->setValue(m_parameters.uniquenessRatio);
    
    m_sgbmLayout->addWidget(m_uniquenessRatioLabel, row, 0);
    m_sgbmLayout->addWidget(m_uniquenessRatioSpin, row, 1);
    m_sgbmLayout->addWidget(m_uniquenessRatioSlider, row, 2);
    
    m_mainLayout->addWidget(m_sgbmGroup);
}

void ParameterPanel::setupPostProcessingGroup()
{
    m_postProcessGroup = new QGroupBox("Post-processing");
    m_postProcessLayout = new QVBoxLayout(m_postProcessGroup);
    
    m_enableSpeckleFilterCheck = new QCheckBox("Enable Speckle Filter");
    m_enableSpeckleFilterCheck->setChecked(m_parameters.enableSpeckleFilter);
    m_postProcessLayout->addWidget(m_enableSpeckleFilterCheck);
    
    m_enableMedianFilterCheck = new QCheckBox("Enable Median Filter");
    m_enableMedianFilterCheck->setChecked(m_parameters.enableMedianFilter);
    m_postProcessLayout->addWidget(m_enableMedianFilterCheck);
    
    m_mainLayout->addWidget(m_postProcessGroup);
}

void ParameterPanel::setupPointCloudGroup()
{
    m_pointCloudGroup = new QGroupBox("Point Cloud");
    m_pointCloudLayout = new QGridLayout(m_pointCloudGroup);
    
    int row = 0;
    
    // Scale Factor
    m_scaleFactorLabel = new QLabel("Scale Factor:");
    m_scaleFactorSpin = new QDoubleSpinBox();
    m_scaleFactorSpin->setRange(0.1, 10.0);
    m_scaleFactorSpin->setSingleStep(0.1);
    m_scaleFactorSpin->setValue(m_parameters.scaleFactor);
    
    m_pointCloudLayout->addWidget(m_scaleFactorLabel, row, 0);
    m_pointCloudLayout->addWidget(m_scaleFactorSpin, row, 1);
    row++;
    
    // Max Depth
    m_maxDepthLabel = new QLabel("Max Depth (m):");
    m_maxDepthSpin = new QDoubleSpinBox();
    m_maxDepthSpin->setRange(1.0, 100.0);
    m_maxDepthSpin->setSingleStep(1.0);
    m_maxDepthSpin->setValue(m_parameters.maxDepth);
    
    m_pointCloudLayout->addWidget(m_maxDepthLabel, row, 0);
    m_pointCloudLayout->addWidget(m_maxDepthSpin, row, 1);
    row++;
    
    // Options
    m_enableColorMappingCheck = new QCheckBox("Enable Color Mapping");
    m_enableColorMappingCheck->setChecked(m_parameters.enableColorMapping);
    m_pointCloudLayout->addWidget(m_enableColorMappingCheck, row, 0, 1, 2);
    row++;
    
    m_enableFilteringCheck = new QCheckBox("Enable Filtering");
    m_enableFilteringCheck->setChecked(m_parameters.enableFiltering);
    m_pointCloudLayout->addWidget(m_enableFilteringCheck, row, 0, 1, 2);
    
    m_mainLayout->addWidget(m_pointCloudGroup);
}

void ParameterPanel::setupPresets()
{
    m_buttonLayout = new QHBoxLayout();
    
    m_presetCombo = new QComboBox();
    m_presetCombo->addItems({"Custom", "Fast", "Accurate", "High Quality"});
    
    m_resetButton = new QPushButton("Reset");
    
    m_buttonLayout->addWidget(new QLabel("Preset:"));
    m_buttonLayout->addWidget(m_presetCombo);
    m_buttonLayout->addWidget(m_resetButton);
    
    m_mainLayout->addLayout(m_buttonLayout);
}

void ParameterPanel::connectSignals()
{
    // SGBM parameters
    connect(m_minDisparitySpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ParameterPanel::onParameterChanged);
    connect(m_minDisparitySlider, &QSlider::valueChanged,
            m_minDisparitySpin, &QSpinBox::setValue);
    connect(m_minDisparitySpin, QOverload<int>::of(&QSpinBox::valueChanged),
            m_minDisparitySlider, &QSlider::setValue);
    
    connect(m_numDisparitiesSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ParameterPanel::onParameterChanged);
    connect(m_numDisparitiesSlider, &QSlider::valueChanged,
            m_numDisparitiesSpin, &QSpinBox::setValue);
    connect(m_numDisparitiesSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            m_numDisparitiesSlider, &QSlider::setValue);
    
    // Post-processing
    connect(m_enableSpeckleFilterCheck, &QCheckBox::toggled,
            this, &ParameterPanel::onParameterChanged);
    connect(m_enableMedianFilterCheck, &QCheckBox::toggled,
            this, &ParameterPanel::onParameterChanged);
    
    // Point cloud
    connect(m_scaleFactorSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ParameterPanel::onParameterChanged);
    connect(m_enableColorMappingCheck, &QCheckBox::toggled,
            this, &ParameterPanel::onParameterChanged);
    connect(m_enableFilteringCheck, &QCheckBox::toggled,
            this, &ParameterPanel::onParameterChanged);
    
    // Buttons
    connect(m_resetButton, &QPushButton::clicked,
            this, &ParameterPanel::onResetClicked);
    connect(m_presetCombo, &QComboBox::currentTextChanged,
            this, &ParameterPanel::onPresetChanged);
}

void ParameterPanel::onParameterChanged()
{
    // Update parameter structure
    m_parameters.minDisparity = m_minDisparitySpin->value();
    m_parameters.numDisparities = m_numDisparitiesSpin->value();
    m_parameters.blockSize = m_blockSizeSpin->value();
    m_parameters.P1 = m_p1Spin->value();
    m_parameters.P2 = m_p2Spin->value();
    m_parameters.uniquenessRatio = m_uniquenessRatioSpin->value();
    
    m_parameters.enableSpeckleFilter = m_enableSpeckleFilterCheck->isChecked();
    m_parameters.enableMedianFilter = m_enableMedianFilterCheck->isChecked();
    
    m_parameters.scaleFactor = m_scaleFactorSpin->value();
    m_parameters.maxDepth = m_maxDepthSpin->value();
    m_parameters.enableColorMapping = m_enableColorMappingCheck->isChecked();
    m_parameters.enableFiltering = m_enableFilteringCheck->isChecked();
    
    // Set to custom preset
    m_presetCombo->setCurrentText("Custom");
    
    emit parametersChanged();
}

void ParameterPanel::onResetClicked()
{
    m_parameters = StereoParameters(); // Reset to defaults
    updateUI();
    emit parametersChanged();
}

void ParameterPanel::onPresetChanged(const QString &preset)
{
    if (preset != "Custom") {
        applyPreset(preset);
        updateUI();
        emit parametersChanged();
    }
}

void ParameterPanel::applyPreset(const QString &preset)
{
    if (preset == "Fast") {
        m_parameters.blockSize = 5;
        m_parameters.numDisparities = 32;
        m_parameters.P1 = 8;
        m_parameters.P2 = 16;
        m_parameters.enableSpeckleFilter = false;
        m_parameters.enableMedianFilter = false;
    } else if (preset == "Accurate") {
        m_parameters.blockSize = 9;
        m_parameters.numDisparities = 64;
        m_parameters.P1 = 8;
        m_parameters.P2 = 64;
        m_parameters.enableSpeckleFilter = true;
        m_parameters.enableMedianFilter = true;
    } else if (preset == "High Quality") {
        m_parameters.blockSize = 15;
        m_parameters.numDisparities = 128;
        m_parameters.P1 = 16;
        m_parameters.P2 = 128;
        m_parameters.enableSpeckleFilter = true;
        m_parameters.enableMedianFilter = true;
    }
}

void ParameterPanel::updateUI()
{
    // Block signals to prevent recursive calls
    m_minDisparitySpin->blockSignals(true);
    m_minDisparitySpin->setValue(m_parameters.minDisparity);
    m_minDisparitySpin->blockSignals(false);
    m_minDisparitySlider->setValue(m_parameters.minDisparity);
    
    m_numDisparitiesSpin->blockSignals(true);
    m_numDisparitiesSpin->setValue(m_parameters.numDisparities);
    m_numDisparitiesSpin->blockSignals(false);
    m_numDisparitiesSlider->setValue(m_parameters.numDisparities);
    
    m_enableSpeckleFilterCheck->setChecked(m_parameters.enableSpeckleFilter);
    m_enableMedianFilterCheck->setChecked(m_parameters.enableMedianFilter);
    
    m_scaleFactorSpin->setValue(m_parameters.scaleFactor);
    m_maxDepthSpin->setValue(m_parameters.maxDepth);
    m_enableColorMappingCheck->setChecked(m_parameters.enableColorMapping);
    m_enableFilteringCheck->setChecked(m_parameters.enableFiltering);
}

void ParameterPanel::setParameters(const StereoParameters &params)
{
    m_parameters = params;
    updateUI();
}

void ParameterPanel::loadSettings()
{
    m_parameters.minDisparity = m_settings->value("sgbm/minDisparity", 0).toInt();
    m_parameters.numDisparities = m_settings->value("sgbm/numDisparities", 64).toInt();
    m_parameters.blockSize = m_settings->value("sgbm/blockSize", 9).toInt();
    m_parameters.enableSpeckleFilter = m_settings->value("post/enableSpeckleFilter", true).toBool();
    m_parameters.enableMedianFilter = m_settings->value("post/enableMedianFilter", true).toBool();
    m_parameters.scaleFactor = m_settings->value("pointcloud/scaleFactor", 1.0).toDouble();
    m_parameters.maxDepth = m_settings->value("pointcloud/maxDepth", 10.0).toDouble();
    
    updateUI();
}

void ParameterPanel::saveSettings()
{
    m_settings->setValue("sgbm/minDisparity", m_parameters.minDisparity);
    m_settings->setValue("sgbm/numDisparities", m_parameters.numDisparities);
    m_settings->setValue("sgbm/blockSize", m_parameters.blockSize);
    m_settings->setValue("post/enableSpeckleFilter", m_parameters.enableSpeckleFilter);
    m_settings->setValue("post/enableMedianFilter", m_parameters.enableMedianFilter);
    m_settings->setValue("pointcloud/scaleFactor", m_parameters.scaleFactor);
    m_settings->setValue("pointcloud/maxDepth", m_parameters.maxDepth);
}

void ParameterPanel::resetToDefaults()
{
    onResetClicked();
}

#include "parameter_panel.moc"
