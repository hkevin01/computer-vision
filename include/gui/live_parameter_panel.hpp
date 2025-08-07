#pragma once

#include <QLabel>
#include <QProgressBar>
#include <QTime>
#include <QTimer>
#include <opencv2/opencv.hpp>

#include "gui/parameter_panel.hpp"

namespace stereo_vision::gui {

/**
 * Enhanced parameter panel with real-time stereo matching preview
 * Extends the basic ParameterPanel with live preview capabilities
 */
class LiveParameterPanel : public ParameterPanel {
    Q_OBJECT

  public:
    explicit LiveParameterPanel(QWidget* parent = nullptr);
    ~LiveParameterPanel();

    // Set stereo images for live preview
    void setStereoImages(const cv::Mat& leftImage, const cv::Mat& rightImage);
    void clearImages();

    // Live preview control
    void enableLivePreview(bool enable);
    bool isLivePreviewEnabled() const {
        return m_livePreviewEnabled;
    }

    // Preview update rate (milliseconds)
    void setPreviewUpdateRate(int ms);
    int getPreviewUpdateRate() const {
        return m_previewTimer->interval();
    }

  signals:
    void disparityMapUpdated(const cv::Mat& disparityMap);
    void previewProcessingStarted();
    void previewProcessingFinished();

  private slots:
    void onParameterChangedLive();
    void updateLivePreview();
    void onLivePreviewToggled(bool enabled);
    void onPreviewRateChanged(int rate);

  private:
    void setupLivePreviewUI();
    void processDisparityMap();
    void updatePreviewStatus(const QString& status);

    // Live preview functionality
    bool m_livePreviewEnabled;
    QTimer* m_previewTimer;
    bool m_previewUpdatePending;

    // Stereo images for preview
    cv::Mat m_leftImage;
    cv::Mat m_rightImage;
    cv::Mat m_currentDisparity;

    // Stereo matcher for live preview
    cv::Ptr<cv::StereoSGBM> m_stereoMatcher;

    // Live preview UI components
    QGroupBox* m_livePreviewGroup;
    QVBoxLayout* m_livePreviewLayout;
    QCheckBox* m_enableLivePreviewCheck;
    QLabel* m_previewRateLabel;
    QSlider* m_previewRateSlider;
    QSpinBox* m_previewRateSpin;
    QLabel* m_previewStatusLabel;
    QProgressBar* m_previewProgressBar;

    // Preview statistics
    QLabel* m_fpsLabel;
    QLabel* m_processingTimeLabel;
    QTimer* m_fpsTimer;
    int m_frameCount;
    QTime m_lastFpsUpdate;

    // Parameter validation indicators
    QLabel* m_validationStatusLabel;

    // Default parameter values for reset
    static const StereoParameters DEFAULT_PARAMETERS;
};

}  // namespace stereo_vision::gui
