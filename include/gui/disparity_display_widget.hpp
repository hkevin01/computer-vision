#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWidget>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

/**
 * Widget for displaying disparity maps with various visualization options
 */
class DisparityDisplayWidget : public QWidget {
    Q_OBJECT

  public:
    enum ColorMap {
        GRAY = 0,
        JET = cv::COLORMAP_JET,
        HOT = cv::COLORMAP_HOT,
        RAINBOW = cv::COLORMAP_RAINBOW,
        PLASMA = cv::COLORMAP_PLASMA,
        VIRIDIS = cv::COLORMAP_VIRIDIS
    };

    explicit DisparityDisplayWidget(QWidget* parent = nullptr);
    ~DisparityDisplayWidget();

    // Set the disparity map to display
    void setDisparityMap(const cv::Mat& disparityMap);
    void clearDisplay();

    // Display options
    void setColorMap(ColorMap colorMap);
    ColorMap getColorMap() const {
        return m_currentColorMap;
    }

    void setAutoScale(bool autoScale);
    bool isAutoScale() const {
        return m_autoScale;
    }

    void setScaleRange(double minValue, double maxValue);
    std::pair<double, double> getScaleRange() const;

    // Zoom and pan controls
    void setZoomFactor(double factor);
    double getZoomFactor() const {
        return m_zoomFactor;
    }
    void resetZoom();
    void fitToWindow();

  public slots:
    void onDisparityMapUpdated(const cv::Mat& disparityMap);

  private slots:
    void onColorMapChanged(int index);
    void onAutoScaleToggled(bool enabled);
    void onScaleRangeChanged();
    void onZoomChanged(int value);
    void onResetZoomClicked();
    void onFitToWindowClicked();

  private:
    void setupUI();
    void updateDisplay();
    void updateDisplayImage();
    cv::Mat applyColorMap(const cv::Mat& grayImage);
    QPixmap matToQPixmap(const cv::Mat& mat);

    // Display data
    cv::Mat m_originalDisparity;
    cv::Mat m_displayImage;
    QPixmap m_displayPixmap;

    // Display settings
    ColorMap m_currentColorMap;
    bool m_autoScale;
    double m_scaleMin;
    double m_scaleMax;
    double m_zoomFactor;

    // UI components
    QVBoxLayout* m_mainLayout;

    // Control panel
    QWidget* m_controlPanel;
    QHBoxLayout* m_controlLayout;

    QLabel* m_colorMapLabel;
    QComboBox* m_colorMapCombo;

    QCheckBox* m_autoScaleCheck;

    QLabel* m_scaleMinLabel;
    QSpinBox* m_scaleMinSpin;
    QLabel* m_scaleMaxLabel;
    QSpinBox* m_scaleMaxSpin;

    QLabel* m_zoomLabel;
    QSlider* m_zoomSlider;
    QSpinBox* m_zoomSpin;
    QPushButton* m_resetZoomButton;
    QPushButton* m_fitToWindowButton;

    // Display area
    QScrollArea* m_scrollArea;
    QLabel* m_imageLabel;

    // Status information
    QLabel* m_statusLabel;
    QLabel* m_dimensionsLabel;
    QLabel* m_rangeLabel;
};

}  // namespace stereo_vision::gui
