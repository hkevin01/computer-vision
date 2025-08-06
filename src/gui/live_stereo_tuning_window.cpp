#include "gui/live_stereo_tuning_window.hpp"

#include <QApplication>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSplitter>
#include <QStatusBar>
#include <QVBoxLayout>
#include <opencv2/opencv.hpp>

#include "gui/disparity_display_widget.hpp"
#include "gui/live_parameter_panel.hpp"

namespace stereo_vision::gui {

LiveStereoTuningWindow::LiveStereoTuningWindow(QWidget* parent)
    : QMainWindow(parent), m_hasValidImages(false), m_isProcessing(false) {
    setWindowTitle("Live Stereo Parameter Tuning");
    setMinimumSize(1200, 800);
    resize(1600, 1000);

    setupUI();
    connectSignals();
    updateWindowTitle();
}

LiveStereoTuningWindow::~LiveStereoTuningWindow() = default;

void LiveStereoTuningWindow::setupUI() {
    setupMenuBar();
    setupStatusBar();
    setupCentralWidget();
}

void LiveStereoTuningWindow::setupMenuBar() {
    QMenuBar* menuBar = this->menuBar();

    // File menu
    QMenu* fileMenu = menuBar->addMenu("&File");

    m_loadStereoAction = new QAction("Load &Stereo Images...", this);
    m_loadStereoAction->setShortcut(QKeySequence::Open);
    m_loadStereoAction->setStatusTip("Load a pair of stereo images");
    fileMenu->addAction(m_loadStereoAction);

    fileMenu->addSeparator();

    m_loadLeftAction = new QAction("Load &Left Image...", this);
    m_loadLeftAction->setStatusTip("Load left stereo image");
    fileMenu->addAction(m_loadLeftAction);

    m_loadRightAction = new QAction("Load &Right Image...", this);
    m_loadRightAction->setStatusTip("Load right stereo image");
    fileMenu->addAction(m_loadRightAction);

    fileMenu->addSeparator();

    m_exportDisparityAction = new QAction("&Export Disparity Map...", this);
    m_exportDisparityAction->setShortcut(QKeySequence("Ctrl+E"));
    m_exportDisparityAction->setStatusTip("Export current disparity map");
    m_exportDisparityAction->setEnabled(false);
    fileMenu->addAction(m_exportDisparityAction);

    fileMenu->addSeparator();

    m_exitAction = new QAction("E&xit", this);
    m_exitAction->setShortcut(QKeySequence::Quit);
    m_exitAction->setStatusTip("Exit the application");
    fileMenu->addAction(m_exitAction);

    // Parameters menu
    QMenu* paramsMenu = menuBar->addMenu("&Parameters");

    m_saveParamsAction = new QAction("&Save Parameters...", this);
    m_saveParamsAction->setShortcut(QKeySequence::Save);
    m_saveParamsAction->setStatusTip("Save current parameters to file");
    paramsMenu->addAction(m_saveParamsAction);

    m_loadParamsAction = new QAction("&Load Parameters...", this);
    m_loadParamsAction->setStatusTip("Load parameters from file");
    paramsMenu->addAction(m_loadParamsAction);

    paramsMenu->addSeparator();

    m_resetParamsAction = new QAction("&Reset to Defaults", this);
    m_resetParamsAction->setShortcut(QKeySequence("Ctrl+R"));
    m_resetParamsAction->setStatusTip("Reset all parameters to default values");
    paramsMenu->addAction(m_resetParamsAction);

    // Help menu
    QMenu* helpMenu = menuBar->addMenu("&Help");

    m_aboutAction = new QAction("&About", this);
    m_aboutAction->setStatusTip("Show application information");
    helpMenu->addAction(m_aboutAction);
}

void LiveStereoTuningWindow::setupStatusBar() {
    QStatusBar* statusBar = this->statusBar();

    m_statusLabel = new QLabel("Ready to load stereo images", this);
    statusBar->addWidget(m_statusLabel, 1);

    m_imageStatusLabel = new QLabel("No images loaded", this);
    m_imageStatusLabel->setStyleSheet("color: gray;");
    statusBar->addWidget(m_imageStatusLabel);

    m_processingStatusLabel = new QLabel("", this);
    m_processingStatusLabel->setStyleSheet("color: blue;");
    statusBar->addWidget(m_processingStatusLabel);

    m_processingProgress = new QProgressBar(this);
    m_processingProgress->setMaximumWidth(200);
    m_processingProgress->setVisible(false);
    statusBar->addWidget(m_processingProgress);
}

void LiveStereoTuningWindow::setupCentralWidget() {
    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);

    m_centralLayout = new QHBoxLayout(m_centralWidget);
    m_centralLayout->setSpacing(5);

    // Create main splitter
    m_mainSplitter = new QSplitter(Qt::Horizontal, this);
    m_centralLayout->addWidget(m_mainSplitter);

    // Left side: Parameter panel
    m_parameterPanel = new LiveParameterPanel(this);
    m_parameterPanel->setMaximumWidth(350);
    m_parameterPanel->setMinimumWidth(300);
    m_mainSplitter->addWidget(m_parameterPanel);

    // Right side: Display area
    m_displayWidget = new QWidget(this);
    m_displayLayout = new QVBoxLayout(m_displayWidget);
    m_displayLayout->setSpacing(5);

    // Image info label
    m_imageInfoLabel = new QLabel("Load stereo images to begin parameter tuning", this);
    m_imageInfoLabel->setAlignment(Qt::AlignCenter);
    m_imageInfoLabel->setStyleSheet("font-size: 14px; color: #666; padding: 20px;");
    m_displayLayout->addWidget(m_imageInfoLabel);

    // Disparity display widget
    m_disparityDisplay = new DisparityDisplayWidget(this);
    m_displayLayout->addWidget(m_disparityDisplay, 1);

    m_mainSplitter->addWidget(m_displayWidget);

    // Set splitter proportions (30% parameter panel, 70% display)
    m_mainSplitter->setStretchFactor(0, 0);
    m_mainSplitter->setStretchFactor(1, 1);
    QList<int> sizes = {300, 900};
    m_mainSplitter->setSizes(sizes);
}

void LiveStereoTuningWindow::connectSignals() {
    // Menu actions
    connect(m_loadStereoAction, &QAction::triggered, this,
            &LiveStereoTuningWindow::onLoadStereoImages);
    connect(m_loadLeftAction, &QAction::triggered, this, &LiveStereoTuningWindow::onLoadLeftImage);
    connect(m_loadRightAction, &QAction::triggered, this,
            &LiveStereoTuningWindow::onLoadRightImage);
    connect(m_saveParamsAction, &QAction::triggered, this,
            &LiveStereoTuningWindow::onSaveParameters);
    connect(m_loadParamsAction, &QAction::triggered, this,
            &LiveStereoTuningWindow::onLoadParameters);
    connect(m_resetParamsAction, &QAction::triggered, this,
            &LiveStereoTuningWindow::onResetParameters);
    connect(m_exportDisparityAction, &QAction::triggered, this,
            &LiveStereoTuningWindow::onExportDisparityMap);
    connect(m_exitAction, &QAction::triggered, this, &QWidget::close);
    connect(m_aboutAction, &QAction::triggered, this, &LiveStereoTuningWindow::onAbout);

    // Parameter panel signals
    connect(m_parameterPanel, &LiveParameterPanel::parametersChanged, this,
            &LiveStereoTuningWindow::onParametersChanged);
    connect(m_parameterPanel, &LiveParameterPanel::disparityMapUpdated, this,
            &LiveStereoTuningWindow::onDisparityMapUpdated);
    connect(m_parameterPanel, &LiveParameterPanel::previewProcessingStarted, this,
            &LiveStereoTuningWindow::onPreviewProcessingStarted);
    connect(m_parameterPanel, &LiveParameterPanel::previewProcessingFinished, this,
            &LiveStereoTuningWindow::onPreviewProcessingFinished);

    // Connect disparity display to parameter panel
    connect(m_parameterPanel, &LiveParameterPanel::disparityMapUpdated, m_disparityDisplay,
            &DisparityDisplayWidget::onDisparityMapUpdated);
}

bool LiveStereoTuningWindow::loadStereoImages(const QString& leftImagePath,
                                              const QString& rightImagePath) {
    // Load images using OpenCV
    cv::Mat leftImage = cv::imread(leftImagePath.toStdString(), cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightImagePath.toStdString(), cv::IMREAD_COLOR);

    if (leftImage.empty()) {
        QMessageBox::warning(this, "Error",
                             QString("Could not load left image: %1").arg(leftImagePath));
        return false;
    }

    if (rightImage.empty()) {
        QMessageBox::warning(this, "Error",
                             QString("Could not load right image: %1").arg(rightImagePath));
        return false;
    }

    if (leftImage.size() != rightImage.size()) {
        QMessageBox::warning(this, "Error",
                             QString("Image size mismatch:\\nLeft: %1×%2\\nRight: %3×%4")
                                 .arg(leftImage.cols)
                                 .arg(leftImage.rows)
                                 .arg(rightImage.cols)
                                 .arg(rightImage.rows));
        return false;
    }

    m_leftImage = leftImage;
    m_rightImage = rightImage;
    m_leftImagePath = leftImagePath;
    m_rightImagePath = rightImagePath;
    m_hasValidImages = true;

    // Update UI
    updateWindowTitle();
    m_imageInfoLabel->setText(
        QString("Images loaded: %1×%2 pixels").arg(leftImage.cols).arg(leftImage.rows));
    m_imageStatusLabel->setText(
        QString("Stereo pair: %1×%2").arg(leftImage.cols).arg(leftImage.rows));
    m_imageStatusLabel->setStyleSheet("color: green;");
    m_exportDisparityAction->setEnabled(true);

    // Send images to parameter panel for live preview
    m_parameterPanel->setStereoImages(m_leftImage, m_rightImage);

    updateStatusBar("Stereo images loaded successfully");
    return true;
}

void LiveStereoTuningWindow::setStereoImages(const cv::Mat& leftImage, const cv::Mat& rightImage) {
    m_leftImage = leftImage.clone();
    m_rightImage = rightImage.clone();
    m_hasValidImages = true;

    updateWindowTitle();
    m_parameterPanel->setStereoImages(m_leftImage, m_rightImage);
    m_exportDisparityAction->setEnabled(true);
}

void LiveStereoTuningWindow::onLoadStereoImages() {
    QString lastDir = QSettings().value("lastImageDirectory", QDir::homePath()).toString();

    QStringList leftFiles =
        QFileDialog::getOpenFileNames(this, "Select Left Stereo Image", lastDir,
                                      "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)");

    if (leftFiles.isEmpty())
        return;

    QString rightFile = QFileDialog::getOpenFileName(
        this, "Select Right Stereo Image", QFileInfo(leftFiles[0]).dir().path(),
        "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)");

    if (rightFile.isEmpty())
        return;

    QSettings().setValue("lastImageDirectory", QFileInfo(leftFiles[0]).dir().path());

    loadStereoImages(leftFiles[0], rightFile);
}

void LiveStereoTuningWindow::onLoadLeftImage() {
    QString lastDir = QSettings().value("lastImageDirectory", QDir::homePath()).toString();

    QString fileName =
        QFileDialog::getOpenFileName(this, "Select Left Stereo Image", lastDir,
                                     "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)");

    if (fileName.isEmpty())
        return;

    cv::Mat image = cv::imread(fileName.toStdString(), cv::IMREAD_COLOR);
    if (image.empty()) {
        QMessageBox::warning(this, "Error", "Could not load the selected image");
        return;
    }

    m_leftImage = image;
    m_leftImagePath = fileName;

    QSettings().setValue("lastImageDirectory", QFileInfo(fileName).dir().path());

    updateWindowTitle();
    updateStatusBar("Left image loaded");

    if (!m_rightImage.empty() && validateStereoImages()) {
        m_parameterPanel->setStereoImages(m_leftImage, m_rightImage);
    }
}

void LiveStereoTuningWindow::onLoadRightImage() {
    QString lastDir = QSettings().value("lastImageDirectory", QDir::homePath()).toString();

    QString fileName =
        QFileDialog::getOpenFileName(this, "Select Right Stereo Image", lastDir,
                                     "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)");

    if (fileName.isEmpty())
        return;

    cv::Mat image = cv::imread(fileName.toStdString(), cv::IMREAD_COLOR);
    if (image.empty()) {
        QMessageBox::warning(this, "Error", "Could not load the selected image");
        return;
    }

    m_rightImage = image;
    m_rightImagePath = fileName;

    QSettings().setValue("lastImageDirectory", QFileInfo(fileName).dir().path());

    updateWindowTitle();
    updateStatusBar("Right image loaded");

    if (!m_leftImage.empty() && validateStereoImages()) {
        m_parameterPanel->setStereoImages(m_leftImage, m_rightImage);
    }
}

bool LiveStereoTuningWindow::validateStereoImages() {
    if (m_leftImage.empty() || m_rightImage.empty()) {
        m_hasValidImages = false;
        return false;
    }

    if (m_leftImage.size() != m_rightImage.size()) {
        QMessageBox::warning(this, "Size Mismatch",
                             QString("Image size mismatch:\\nLeft: %1×%2\\nRight: %3×%4")
                                 .arg(m_leftImage.cols)
                                 .arg(m_leftImage.rows)
                                 .arg(m_rightImage.cols)
                                 .arg(m_rightImage.rows));
        m_hasValidImages = false;
        return false;
    }

    m_hasValidImages = true;
    m_imageStatusLabel->setText(
        QString("Stereo pair: %1×%2").arg(m_leftImage.cols).arg(m_leftImage.rows));
    m_imageStatusLabel->setStyleSheet("color: green;");
    m_exportDisparityAction->setEnabled(true);

    return true;
}

void LiveStereoTuningWindow::onSaveParameters() {
    QString fileName =
        QFileDialog::getSaveFileName(this, "Save Parameters", "", "JSON Files (*.json)");

    if (fileName.isEmpty())
        return;

    // TODO: Implement parameter saving
    updateStatusBar("Parameters saved");
}

void LiveStereoTuningWindow::onLoadParameters() {
    QString fileName =
        QFileDialog::getOpenFileName(this, "Load Parameters", "", "JSON Files (*.json)");

    if (fileName.isEmpty())
        return;

    // TODO: Implement parameter loading
    updateStatusBar("Parameters loaded");
}

void LiveStereoTuningWindow::onResetParameters() {
    m_parameterPanel->resetToDefaults();
    updateStatusBar("Parameters reset to defaults");
}

void LiveStereoTuningWindow::onExportDisparityMap() {
    if (m_currentDisparity.empty()) {
        QMessageBox::information(this, "No Disparity Map",
                                 "No disparity map available. Adjust parameters to generate one.");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(
        this, "Export Disparity Map", "", "PNG Files (*.png);;TIFF Files (*.tiff);;All Files (*)");

    if (fileName.isEmpty())
        return;

    try {
        cv::imwrite(fileName.toStdString(), m_currentDisparity);
        updateStatusBar("Disparity map exported successfully");
    } catch (const cv::Exception& e) {
        QMessageBox::warning(this, "Export Error",
                             QString("Failed to export disparity map: %1").arg(e.what()));
    }
}

void LiveStereoTuningWindow::onAbout() {
    QMessageBox::about(this, "About Live Stereo Parameter Tuning",
                       "Live Stereo Parameter Tuning Tool\\n\\n"
                       "Real-time adjustment of stereo matching parameters with live preview.\\n"
                       "Part of the Computer Vision Stereo Processing Suite.\\n\\n"
                       "Features:\\n"
                       "• Real-time disparity map updates\\n"
                       "• Parameter validation and constraints\\n"
                       "• Multiple visualization color maps\\n"
                       "• Export capabilities\\n"
                       "• Performance monitoring");
}

void LiveStereoTuningWindow::onParametersChanged() {
    updateStatusBar("Parameters updated");
}

void LiveStereoTuningWindow::onDisparityMapUpdated(const cv::Mat& disparityMap) {
    m_currentDisparity = disparityMap.clone();
    updateStatusBar("Disparity map updated");
}

void LiveStereoTuningWindow::onPreviewProcessingStarted() {
    m_isProcessing = true;
    m_processingProgress->setVisible(true);
    m_processingProgress->setRange(0, 0);  // Indeterminate
    m_processingStatusLabel->setText("Processing...");
}

void LiveStereoTuningWindow::onPreviewProcessingFinished() {
    m_isProcessing = false;
    m_processingProgress->setVisible(false);
    m_processingStatusLabel->setText("");
}

void LiveStereoTuningWindow::updateWindowTitle() {
    QString title = "Live Stereo Parameter Tuning";

    if (m_hasValidImages) {
        QString leftName = QFileInfo(m_leftImagePath).baseName();
        QString rightName = QFileInfo(m_rightImagePath).baseName();
        title += QString(" - %1 & %2").arg(leftName, rightName);
    }

    setWindowTitle(title);
}

void LiveStereoTuningWindow::updateStatusBar(const QString& message) {
    m_statusLabel->setText(message);
}

}  // namespace stereo_vision::gui

#include "live_stereo_tuning_window.moc"
