#include "gui/live_parameter_panel.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QSlider>
#include <QSpinBox>
#include <QTime>
#include <QTimer>
#include <QVBoxLayout>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

// Default parameter values for validation and reset
const StereoParameters LiveParameterPanel::DEFAULT_PARAMETERS = {.minDisparity = 0,
                                                                 .numDisparities = 64,
                                                                 .blockSize = 9,
                                                                 .P1 = 8,
                                                                 .P2 = 32,
                                                                 .disp12MaxDiff = 1,
                                                                 .preFilterCap = 63,
                                                                 .uniquenessRatio = 10,
                                                                 .speckleWindowSize = 100,
                                                                 .speckleRange = 32,
                                                                 .mode = 0,
                                                                 .enableSpeckleFilter = true,
                                                                 .enableMedianFilter = true,
                                                                 .medianKernelSize = 5,
                                                                 .scaleFactor = 1.0,
                                                                 .enableColorMapping = true,
                                                                 .enableFiltering = true,
                                                                 .maxDepth = 10.0,
                                                                 .minDepth = 0.1};

LiveParameterPanel::LiveParameterPanel(QWidget* parent)
    : ParameterPanel(parent), m_livePreviewEnabled(false), m_previewTimer(new QTimer(this)),
      m_previewUpdatePending(false), m_fpsTimer(new QTimer(this)), m_frameCount(0) {
    setupLivePreviewUI();

    // Initialize stereo matcher
    m_stereoMatcher = cv::StereoSGBM::create();

    // Setup timers
    m_previewTimer->setSingleShot(true);
    m_previewTimer->setInterval(100);  // 100ms default update rate
    connect(m_previewTimer, &QTimer::timeout, this, &LiveParameterPanel::updateLivePreview);

    // FPS calculation timer (update every second)
    m_fpsTimer->setInterval(1000);
    connect(m_fpsTimer, &QTimer::timeout, this, [this]() {
        if (m_livePreviewEnabled && m_frameCount > 0) {
            double fps = m_frameCount;
            m_fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
            m_frameCount = 0;
        }
    });
    m_fpsTimer->start();

    // Connect parameter changes to live preview
    connect(this, QOverload<>::of(&ParameterPanel::parametersChanged), this,
            &LiveParameterPanel::onParameterChangedLive);

    m_lastFpsUpdate = QTime::currentTime();
}

LiveParameterPanel::~LiveParameterPanel() {
    m_previewTimer->stop();
    m_fpsTimer->stop();
}

void LiveParameterPanel::setupLivePreviewUI() {
    // Create live preview group box
    m_livePreviewGroup = new QGroupBox("Live Preview Controls", this);
    m_livePreviewLayout = new QVBoxLayout(m_livePreviewGroup);

    // Enable live preview checkbox
    m_enableLivePreviewCheck = new QCheckBox("Enable Live Preview", this);
    m_enableLivePreviewCheck->setChecked(false);
    m_enableLivePreviewCheck->setToolTip(
        "Enable real-time disparity map updates when parameters change");
    connect(m_enableLivePreviewCheck, &QCheckBox::toggled, this,
            &LiveParameterPanel::onLivePreviewToggled);
    m_livePreviewLayout->addWidget(m_enableLivePreviewCheck);

    // Preview update rate controls
    QHBoxLayout* rateLayout = new QHBoxLayout();
    m_previewRateLabel = new QLabel("Update Rate (ms):", this);
    m_previewRateSlider = new QSlider(Qt::Horizontal, this);
    m_previewRateSlider->setRange(50, 1000);
    m_previewRateSlider->setValue(100);
    m_previewRateSlider->setToolTip("Time delay between parameter changes and preview update");

    m_previewRateSpin = new QSpinBox(this);
    m_previewRateSpin->setRange(50, 1000);
    m_previewRateSpin->setValue(100);
    m_previewRateSpin->setSuffix(" ms");

    // Connect rate controls
    connect(m_previewRateSlider, &QSlider::valueChanged, m_previewRateSpin, &QSpinBox::setValue);
    connect(m_previewRateSpin, QOverload<int>::of(&QSpinBox::valueChanged), m_previewRateSlider,
            &QSlider::setValue);
    connect(m_previewRateSlider, &QSlider::valueChanged, this,
            &LiveParameterPanel::onPreviewRateChanged);

    rateLayout->addWidget(m_previewRateLabel);
    rateLayout->addWidget(m_previewRateSlider, 1);
    rateLayout->addWidget(m_previewRateSpin);
    m_livePreviewLayout->addLayout(rateLayout);

    // Preview status and progress
    m_previewStatusLabel = new QLabel("Preview: Disabled", this);
    m_previewStatusLabel->setStyleSheet("color: gray; font-weight: bold;");
    m_livePreviewLayout->addWidget(m_previewStatusLabel);

    m_previewProgressBar = new QProgressBar(this);
    m_previewProgressBar->setVisible(false);
    m_previewProgressBar->setRange(0, 0);  // Indeterminate progress
    m_livePreviewLayout->addWidget(m_previewProgressBar);

    // Performance statistics
    QHBoxLayout* statsLayout = new QHBoxLayout();
    m_fpsLabel = new QLabel("FPS: 0.0", this);
    m_processingTimeLabel = new QLabel("Processing: 0 ms", this);
    m_fpsLabel->setStyleSheet("font-size: 10px; color: #666;");
    m_processingTimeLabel->setStyleSheet("font-size: 10px; color: #666;");

    statsLayout->addWidget(m_fpsLabel);
    statsLayout->addStretch();
    statsLayout->addWidget(m_processingTimeLabel);
    m_livePreviewLayout->addLayout(statsLayout);

    // Parameter validation status
    m_validationStatusLabel = new QLabel("Parameters: Valid ✓", this);
    m_validationStatusLabel->setStyleSheet("font-size: 10px; color: green;");
    m_livePreviewLayout->addWidget(m_validationStatusLabel);

    // Add the live preview group to the main layout
    // Insert it after the existing parameter groups
    layout()->addWidget(m_livePreviewGroup);
}

void LiveParameterPanel::setStereoImages(const cv::Mat& leftImage, const cv::Mat& rightImage) {
    if (leftImage.empty() || rightImage.empty()) {
        clearImages();
        return;
    }

    // Ensure images are the same size
    if (leftImage.size() != rightImage.size()) {
        updatePreviewStatus("Error: Image size mismatch");
        m_validationStatusLabel->setText("Images: Size mismatch ✗");
        m_validationStatusLabel->setStyleSheet("font-size: 10px; color: red;");
        return;
    }

    // Convert to grayscale if needed
    if (leftImage.channels() == 3) {
        cv::cvtColor(leftImage, m_leftImage, cv::COLOR_BGR2GRAY);
    } else {
        m_leftImage = leftImage.clone();
    }

    if (rightImage.channels() == 3) {
        cv::cvtColor(rightImage, m_rightImage, cv::COLOR_BGR2GRAY);
    } else {
        m_rightImage = rightImage.clone();
    }

    updatePreviewStatus("Images loaded successfully");
    m_validationStatusLabel->setText("Images: Loaded ✓");
    m_validationStatusLabel->setStyleSheet("font-size: 10px; color: green;");

    // Trigger initial preview if enabled
    if (m_livePreviewEnabled) {
        onParameterChangedLive();
    }
}

void LiveParameterPanel::clearImages() {
    m_leftImage = cv::Mat();
    m_rightImage = cv::Mat();
    m_currentDisparity = cv::Mat();

    updatePreviewStatus("No images loaded");
    m_validationStatusLabel->setText("Images: None loaded");
    m_validationStatusLabel->setStyleSheet("font-size: 10px; color: gray;");
}

void LiveParameterPanel::enableLivePreview(bool enable) {
    m_enableLivePreviewCheck->setChecked(enable);
}

void LiveParameterPanel::setPreviewUpdateRate(int ms) {
    m_previewRateSlider->setValue(ms);
}

void LiveParameterPanel::onParameterChangedLive() {
    if (!m_livePreviewEnabled || m_leftImage.empty() || m_rightImage.empty()) {
        return;
    }

    // Validate parameters before processing
    StereoParameters params = getParameters();
    bool paramsValid = true;
    QString validationMessage;

    // Validate critical parameters
    if (params.numDisparities <= 0 || params.numDisparities % 16 != 0) {
        paramsValid = false;
        validationMessage = "numDisparities must be > 0 and divisible by 16";
    } else if (params.blockSize < 3 || params.blockSize % 2 == 0) {
        paramsValid = false;
        validationMessage = "blockSize must be odd and >= 3";
    } else if (params.P1 <= 0 || params.P2 <= 0 || params.P2 < params.P1) {
        paramsValid = false;
        validationMessage = "P1 > 0, P2 > 0, and P2 >= P1";
    }

    if (!paramsValid) {
        m_validationStatusLabel->setText(QString("Parameters: %1 ✗").arg(validationMessage));
        m_validationStatusLabel->setStyleSheet("font-size: 10px; color: red;");
        return;
    }

    m_validationStatusLabel->setText("Parameters: Valid ✓");
    m_validationStatusLabel->setStyleSheet("font-size: 10px; color: green;");

    // Schedule preview update with debouncing
    m_previewUpdatePending = true;
    m_previewTimer->start();  // Restart timer for debouncing
}

void LiveParameterPanel::updateLivePreview() {
    if (!m_previewUpdatePending || !m_livePreviewEnabled || m_leftImage.empty() ||
        m_rightImage.empty()) {
        return;
    }

    m_previewUpdatePending = false;
    emit previewProcessingStarted();

    // Show progress indicator
    m_previewProgressBar->setVisible(true);
    updatePreviewStatus("Processing...");

    // Process in a separate thread-like manner using QTimer
    QTimer::singleShot(0, this, &LiveParameterPanel::processDisparityMap);
}

void LiveParameterPanel::processDisparityMap() {
    QTime processStart = QTime::currentTime();

    try {
        // Get current parameters
        StereoParameters params = getParameters();

        // Configure stereo matcher
        m_stereoMatcher->setMinDisparity(params.minDisparity);
        m_stereoMatcher->setNumDisparities(params.numDisparities);
        m_stereoMatcher->setBlockSize(params.blockSize);
        m_stereoMatcher->setP1(params.P1 * params.blockSize * params.blockSize);
        m_stereoMatcher->setP2(params.P2 * params.blockSize * params.blockSize);
        m_stereoMatcher->setDisp12MaxDiff(params.disp12MaxDiff);
        m_stereoMatcher->setPreFilterCap(params.preFilterCap);
        m_stereoMatcher->setUniquenessRatio(params.uniquenessRatio);
        m_stereoMatcher->setSpeckleWindowSize(params.speckleWindowSize);
        m_stereoMatcher->setSpeckleRange(params.speckleRange);
        m_stereoMatcher->setMode(params.mode);

        // Compute disparity map
        cv::Mat disparity16;
        m_stereoMatcher->compute(m_leftImage, m_rightImage, disparity16);

        // Convert to 8-bit for display
        cv::Mat disparity8;
        disparity16.convertTo(disparity8, CV_8UC1, 255.0 / (params.numDisparities * 16.0));

        // Apply post-processing if enabled
        if (params.enableMedianFilter) {
            cv::medianBlur(disparity8, disparity8, params.medianKernelSize);
        }

        m_currentDisparity = disparity8.clone();

        // Calculate processing time
        int processingTime = processStart.msecsTo(QTime::currentTime());
        m_processingTimeLabel->setText(QString("Processing: %1 ms").arg(processingTime));

        // Update FPS counter
        m_frameCount++;

        // Emit signal with result
        emit disparityMapUpdated(m_currentDisparity);

        updatePreviewStatus("Preview updated");

    } catch (const cv::Exception& e) {
        updatePreviewStatus(QString("OpenCV Error: %1").arg(e.what()));
        m_validationStatusLabel->setText("Processing: Error ✗");
        m_validationStatusLabel->setStyleSheet("font-size: 10px; color: red;");
    } catch (const std::exception& e) {
        updatePreviewStatus(QString("Error: %1").arg(e.what()));
        m_validationStatusLabel->setText("Processing: Error ✗");
        m_validationStatusLabel->setStyleSheet("font-size: 10px; color: red;");
    }

    // Hide progress indicator
    m_previewProgressBar->setVisible(false);
    emit previewProcessingFinished();
}

void LiveParameterPanel::onLivePreviewToggled(bool enabled) {
    m_livePreviewEnabled = enabled;

    if (enabled) {
        updatePreviewStatus("Live preview enabled");
        m_previewStatusLabel->setStyleSheet("color: green; font-weight: bold;");

        // Trigger immediate update if images are loaded
        if (!m_leftImage.empty() && !m_rightImage.empty()) {
            onParameterChangedLive();
        }
    } else {
        updatePreviewStatus("Live preview disabled");
        m_previewStatusLabel->setStyleSheet("color: gray; font-weight: bold;");
        m_previewTimer->stop();
        m_previewProgressBar->setVisible(false);
        m_frameCount = 0;
        m_fpsLabel->setText("FPS: 0.0");
    }
}

void LiveParameterPanel::onPreviewRateChanged(int rate) {
    m_previewTimer->setInterval(rate);
    updatePreviewStatus(QString("Update rate: %1 ms").arg(rate));
}

void LiveParameterPanel::updatePreviewStatus(const QString& status) {
    m_previewStatusLabel->setText(QString("Preview: %1").arg(status));
}

}  // namespace stereo_vision::gui
