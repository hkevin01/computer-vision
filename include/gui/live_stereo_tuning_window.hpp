#pragma once

#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QProgressBar>
#include <QPushButton>
#include <QSplitter>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

class LiveParameterPanel;
class DisparityDisplayWidget;

/**
 * Main window for real-time stereo parameter tuning
 * Provides live preview of disparity maps as parameters change
 */
class LiveStereoTuningWindow : public QMainWindow {
    Q_OBJECT

  public:
    explicit LiveStereoTuningWindow(QWidget* parent = nullptr);
    ~LiveStereoTuningWindow();

    // Load stereo image pair for tuning
    bool loadStereoImages(const QString& leftImagePath, const QString& rightImagePath);
    void setStereoImages(const cv::Mat& leftImage, const cv::Mat& rightImage);

    // Get current parameter values
    struct StereoParameters getCurrentParameters() const;

  public slots:
    void onLoadStereoImages();
    void onLoadLeftImage();
    void onLoadRightImage();
    void onSaveParameters();
    void onLoadParameters();
    void onResetParameters();
    void onExportDisparityMap();
    void onAbout();

  private slots:
    void onParametersChanged();
    void onDisparityMapUpdated(const cv::Mat& disparityMap);
    void onPreviewProcessingStarted();
    void onPreviewProcessingFinished();

  private:
    void setupUI();
    void setupMenuBar();
    void setupStatusBar();
    void setupCentralWidget();
    void connectSignals();
    void updateWindowTitle();
    void updateStatusBar(const QString& message);
    bool validateStereoImages();

    // UI components
    QWidget* m_centralWidget;
    QHBoxLayout* m_centralLayout;
    QSplitter* m_mainSplitter;

    // Parameter panel (left side)
    LiveParameterPanel* m_parameterPanel;

    // Display area (right side)
    QWidget* m_displayWidget;
    QVBoxLayout* m_displayLayout;
    QLabel* m_imageInfoLabel;
    DisparityDisplayWidget* m_disparityDisplay;

    // Menu actions
    QAction* m_loadStereoAction;
    QAction* m_loadLeftAction;
    QAction* m_loadRightAction;
    QAction* m_saveParamsAction;
    QAction* m_loadParamsAction;
    QAction* m_resetParamsAction;
    QAction* m_exportDisparityAction;
    QAction* m_exitAction;
    QAction* m_aboutAction;

    // Status bar components
    QLabel* m_statusLabel;
    QLabel* m_imageStatusLabel;
    QLabel* m_processingStatusLabel;
    QProgressBar* m_processingProgress;

    // Current data
    cv::Mat m_leftImage;
    cv::Mat m_rightImage;
    cv::Mat m_currentDisparity;
    QString m_leftImagePath;
    QString m_rightImagePath;

    // State tracking
    bool m_hasValidImages;
    bool m_isProcessing;
};

}  // namespace stereo_vision::gui
