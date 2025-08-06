#include "gui/disparity_display_widget.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QScrollArea>
#include <QSize>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

DisparityDisplayWidget::DisparityDisplayWidget(QWidget* parent)
    : QWidget(parent), m_currentColorMap(ColorMap::JET), m_autoScale(true), m_scaleMin(0.0),
      m_scaleMax(255.0), m_zoomFactor(1.0) {
    setupUI();
}

DisparityDisplayWidget::~DisparityDisplayWidget() = default;

void DisparityDisplayWidget::setupUI() {
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setSpacing(5);

    // Control panel
    m_controlPanel = new QWidget(this);
    m_controlLayout = new QHBoxLayout(m_controlPanel);
    m_controlLayout->setSpacing(10);

    // Color map selection
    m_colorMapLabel = new QLabel("Color Map:", this);
    m_colorMapCombo = new QComboBox(this);
    m_colorMapCombo->addItem("Grayscale", static_cast<int>(ColorMap::GRAY));
    m_colorMapCombo->addItem("Jet", static_cast<int>(ColorMap::JET));
    m_colorMapCombo->addItem("Hot", static_cast<int>(ColorMap::HOT));
    m_colorMapCombo->addItem("Rainbow", static_cast<int>(ColorMap::RAINBOW));
    m_colorMapCombo->addItem("Plasma", static_cast<int>(ColorMap::PLASMA));
    m_colorMapCombo->addItem("Viridis", static_cast<int>(ColorMap::VIRIDIS));
    m_colorMapCombo->setCurrentIndex(1);  // Default to Jet
    connect(m_colorMapCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &DisparityDisplayWidget::onColorMapChanged);

    // Auto-scale checkbox
    m_autoScaleCheck = new QCheckBox("Auto Scale", this);
    m_autoScaleCheck->setChecked(true);
    connect(m_autoScaleCheck, &QCheckBox::toggled, this,
            &DisparityDisplayWidget::onAutoScaleToggled);

    // Scale range controls
    m_scaleMinLabel = new QLabel("Min:", this);
    m_scaleMinSpin = new QSpinBox(this);
    m_scaleMinSpin->setRange(0, 65535);
    m_scaleMinSpin->setValue(0);
    m_scaleMinSpin->setEnabled(false);  // Disabled when auto-scale is on
    connect(m_scaleMinSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &DisparityDisplayWidget::onScaleRangeChanged);

    m_scaleMaxLabel = new QLabel("Max:", this);
    m_scaleMaxSpin = new QSpinBox(this);
    m_scaleMaxSpin->setRange(0, 65535);
    m_scaleMaxSpin->setValue(255);
    m_scaleMaxSpin->setEnabled(false);  // Disabled when auto-scale is on
    connect(m_scaleMaxSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &DisparityDisplayWidget::onScaleRangeChanged);

    // Zoom controls
    m_zoomLabel = new QLabel("Zoom:", this);
    m_zoomSlider = new QSlider(Qt::Horizontal, this);
    m_zoomSlider->setRange(10, 500);  // 10% to 500%
    m_zoomSlider->setValue(100);
    m_zoomSlider->setToolTip("Zoom level (%)");

    m_zoomSpin = new QSpinBox(this);
    m_zoomSpin->setRange(10, 500);
    m_zoomSpin->setValue(100);
    m_zoomSpin->setSuffix("%");

    // Connect zoom controls
    connect(m_zoomSlider, &QSlider::valueChanged, m_zoomSpin, &QSpinBox::setValue);
    connect(m_zoomSpin, QOverload<int>::of(&QSpinBox::valueChanged), m_zoomSlider,
            &QSlider::setValue);
    connect(m_zoomSlider, &QSlider::valueChanged, this, &DisparityDisplayWidget::onZoomChanged);

    // Zoom buttons
    m_resetZoomButton = new QPushButton("Reset", this);
    m_resetZoomButton->setToolTip("Reset zoom to 100%");
    connect(m_resetZoomButton, &QPushButton::clicked, this,
            &DisparityDisplayWidget::onResetZoomClicked);

    m_fitToWindowButton = new QPushButton("Fit", this);
    m_fitToWindowButton->setToolTip("Fit image to window");
    connect(m_fitToWindowButton, &QPushButton::clicked, this,
            &DisparityDisplayWidget::onFitToWindowClicked);

    // Add controls to layout
    m_controlLayout->addWidget(m_colorMapLabel);
    m_controlLayout->addWidget(m_colorMapCombo);
    m_controlLayout->addWidget(m_autoScaleCheck);
    m_controlLayout->addWidget(m_scaleMinLabel);
    m_controlLayout->addWidget(m_scaleMinSpin);
    m_controlLayout->addWidget(m_scaleMaxLabel);
    m_controlLayout->addWidget(m_scaleMaxSpin);
    m_controlLayout->addStretch();
    m_controlLayout->addWidget(m_zoomLabel);
    m_controlLayout->addWidget(m_zoomSlider);
    m_controlLayout->addWidget(m_zoomSpin);
    m_controlLayout->addWidget(m_resetZoomButton);
    m_controlLayout->addWidget(m_fitToWindowButton);

    m_mainLayout->addWidget(m_controlPanel);

    // Display area with scroll
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(false);
    m_scrollArea->setAlignment(Qt::AlignCenter);
    m_scrollArea->setStyleSheet("QScrollArea { border: 1px solid gray; }");

    m_imageLabel = new QLabel(this);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setStyleSheet("QLabel { background-color: #f0f0f0; }");
    m_imageLabel->setText("No disparity map loaded");
    m_imageLabel->setMinimumSize(300, 200);

    m_scrollArea->setWidget(m_imageLabel);
    m_mainLayout->addWidget(m_scrollArea, 1);

    // Status bar
    QHBoxLayout* statusLayout = new QHBoxLayout();
    m_statusLabel = new QLabel("Status: Ready", this);
    m_dimensionsLabel = new QLabel("", this);
    m_rangeLabel = new QLabel("", this);

    m_statusLabel->setStyleSheet("font-size: 10px; color: #666;");
    m_dimensionsLabel->setStyleSheet("font-size: 10px; color: #666;");
    m_rangeLabel->setStyleSheet("font-size: 10px; color: #666;");

    statusLayout->addWidget(m_statusLabel);
    statusLayout->addStretch();
    statusLayout->addWidget(m_dimensionsLabel);
    statusLayout->addStretch();
    statusLayout->addWidget(m_rangeLabel);

    m_mainLayout->addLayout(statusLayout);
}

void DisparityDisplayWidget::setDisparityMap(const cv::Mat& disparityMap) {
    if (disparityMap.empty()) {
        clearDisplay();
        return;
    }

    m_originalDisparity = disparityMap.clone();

    // Update status information
    m_dimensionsLabel->setText(
        QString("Size: %1×%2").arg(disparityMap.cols).arg(disparityMap.rows));

    // Calculate min/max for range display
    double minVal, maxVal;
    cv::minMaxLoc(disparityMap, &minVal, &maxVal);
    m_rangeLabel->setText(QString("Range: %1 - %2").arg(minVal, 0, 'f', 1).arg(maxVal, 0, 'f', 1));

    // Update auto-scale values if enabled
    if (m_autoScale) {
        m_scaleMin = minVal;
        m_scaleMax = maxVal;
        m_scaleMinSpin->setValue(static_cast<int>(minVal));
        m_scaleMaxSpin->setValue(static_cast<int>(maxVal));
    }

    updateDisplay();
    m_statusLabel->setText("Status: Disparity map loaded");
}

void DisparityDisplayWidget::clearDisplay() {
    m_originalDisparity = cv::Mat();
    m_displayImage = cv::Mat();
    m_imageLabel->setText("No disparity map loaded");
    m_imageLabel->setPixmap(QPixmap());

    m_statusLabel->setText("Status: No data");
    m_dimensionsLabel->setText("");
    m_rangeLabel->setText("");
}

void DisparityDisplayWidget::onDisparityMapUpdated(const cv::Mat& disparityMap) {
    setDisparityMap(disparityMap);
}

void DisparityDisplayWidget::updateDisplay() {
    if (m_originalDisparity.empty()) {
        return;
    }

    updateDisplayImage();

    // Convert to QPixmap and display
    m_displayPixmap = matToQPixmap(m_displayImage);

    // Apply zoom
    if (m_zoomFactor != 1.0) {
        QSize scaledSize = m_displayPixmap.size() * m_zoomFactor;
        m_displayPixmap =
            m_displayPixmap.scaled(scaledSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    m_imageLabel->setPixmap(m_displayPixmap);
    m_imageLabel->resize(m_displayPixmap.size());

    m_statusLabel->setText("Status: Display updated");
}

void DisparityDisplayWidget::updateDisplayImage() {
    if (m_originalDisparity.empty()) {
        return;
    }

    cv::Mat scaledDisparity;

    // Scale the disparity map to 0-255 range
    if (m_autoScale) {
        double minVal, maxVal;
        cv::minMaxLoc(m_originalDisparity, &minVal, &maxVal);
        if (maxVal > minVal) {
            m_originalDisparity.convertTo(scaledDisparity, CV_8UC1, 255.0 / (maxVal - minVal),
                                          -minVal * 255.0 / (maxVal - minVal));
        } else {
            scaledDisparity = cv::Mat::zeros(m_originalDisparity.size(), CV_8UC1);
        }
    } else {
        double scale = 255.0 / (m_scaleMax - m_scaleMin);
        double offset = -m_scaleMin * scale;
        m_originalDisparity.convertTo(scaledDisparity, CV_8UC1, scale, offset);
    }

    // Apply color map
    m_displayImage = applyColorMap(scaledDisparity);
}

cv::Mat DisparityDisplayWidget::applyColorMap(const cv::Mat& grayImage) {
    if (m_currentColorMap == ColorMap::GRAY) {
        cv::Mat result;
        cv::cvtColor(grayImage, result, cv::COLOR_GRAY2BGR);
        return result;
    } else {
        cv::Mat result;
        cv::applyColorMap(grayImage, result, static_cast<int>(m_currentColorMap));
        return result;
    }
}

QPixmap DisparityDisplayWidget::matToQPixmap(const cv::Mat& mat) {
    if (mat.empty()) {
        return QPixmap();
    }

    cv::Mat rgbMat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(mat, rgbMat, cv::COLOR_GRAY2RGB);
    }

    QImage qimg(rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888);
    return QPixmap::fromImage(qimg);
}

void DisparityDisplayWidget::onColorMapChanged(int index) {
    m_currentColorMap = static_cast<ColorMap>(m_colorMapCombo->itemData(index).toInt());
    updateDisplay();
}

void DisparityDisplayWidget::onAutoScaleToggled(bool enabled) {
    m_autoScale = enabled;
    m_scaleMinSpin->setEnabled(!enabled);
    m_scaleMaxSpin->setEnabled(!enabled);

    if (enabled && !m_originalDisparity.empty()) {
        double minVal, maxVal;
        cv::minMaxLoc(m_originalDisparity, &minVal, &maxVal);
        m_scaleMin = minVal;
        m_scaleMax = maxVal;
        m_scaleMinSpin->setValue(static_cast<int>(minVal));
        m_scaleMaxSpin->setValue(static_cast<int>(maxVal));
    }

    updateDisplay();
}

void DisparityDisplayWidget::onScaleRangeChanged() {
    if (!m_autoScale) {
        m_scaleMin = m_scaleMinSpin->value();
        m_scaleMax = m_scaleMaxSpin->value();

        // Ensure max >= min
        if (m_scaleMax < m_scaleMin) {
            m_scaleMax = m_scaleMin + 1;
            m_scaleMaxSpin->setValue(static_cast<int>(m_scaleMax));
        }

        updateDisplay();
    }
}

void DisparityDisplayWidget::onZoomChanged(int value) {
    m_zoomFactor = value / 100.0;
    updateDisplay();
}

void DisparityDisplayWidget::onResetZoomClicked() {
    m_zoomSlider->setValue(100);
}

void DisparityDisplayWidget::onFitToWindowClicked() {
    if (m_originalDisparity.empty()) {
        return;
    }

    // Calculate zoom factor to fit the image in the scroll area
    QSize scrollAreaSize = m_scrollArea->size();
    QSize imageSize = m_originalDisparity.size();

    double scaleX =
        static_cast<double>(scrollAreaSize.width() - 20) / imageSize.width();  // 20px margin
    double scaleY = static_cast<double>(scrollAreaSize.height() - 20) / imageSize.height();
    double scale = qMin(scaleX, scaleY);

    int zoomPercent = static_cast<int>(scale * 100);
    zoomPercent = qMax(10, qMin(500, zoomPercent));  // Clamp to slider range

    m_zoomSlider->setValue(zoomPercent);
}

void DisparityDisplayWidget::setColorMap(ColorMap colorMap) {
    for (int i = 0; i < m_colorMapCombo->count(); ++i) {
        if (m_colorMapCombo->itemData(i).toInt() == static_cast<int>(colorMap)) {
            m_colorMapCombo->setCurrentIndex(i);
            break;
        }
    }
}

void DisparityDisplayWidget::setAutoScale(bool autoScale) {
    m_autoScaleCheck->setChecked(autoScale);
}

void DisparityDisplayWidget::setScaleRange(double minValue, double maxValue) {
    m_scaleMinSpin->setValue(static_cast<int>(minValue));
    m_scaleMaxSpin->setValue(static_cast<int>(maxValue));
}

std::pair<double, double> DisparityDisplayWidget::getScaleRange() const {
    return {m_scaleMin, m_scaleMax};
}

void DisparityDisplayWidget::setZoomFactor(double factor) {
    int zoomPercent = static_cast<int>(factor * 100);
    m_zoomSlider->setValue(zoomPercent);
}

void DisparityDisplayWidget::resetZoom() {
    onResetZoomClicked();
}

void DisparityDisplayWidget::fitToWindow() {
    onFitToWindowClicked();
}

}  // namespace stereo_vision::gui

#include "disparity_display_widget.moc"
