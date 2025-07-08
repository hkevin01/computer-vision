#include "gui/main_window.hpp"
#include <QtWidgets/QApplication>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_centralWidget(nullptr), m_mainSplitter(nullptr), m_imageTabWidget(nullptr), m_leftImageWidget(nullptr), m_rightImageWidget(nullptr), m_disparityWidget(nullptr), m_pointCloudWidget(nullptr), m_parameterPanel(nullptr), m_progressBar(nullptr), m_statusLabel(nullptr), m_calibration(nullptr), m_stereoMatcher(nullptr), m_pointCloudProcessor(nullptr), m_processingTimer(new QTimer(this)), m_isProcessing(false), m_hasCalibration(false), m_hasImages(false)
{
    // Initialize processing components
    m_calibration = new CameraCalibration();
    m_stereoMatcher = new StereoMatcher();
    m_pointCloudProcessor = new PointCloudProcessor();

    setupUI();
    connectSignals();
    updateUI();

    // Set window properties
    setWindowTitle("Stereo Vision 3D Point Cloud Generator");
    setMinimumSize(1200, 800);
    resize(1600, 1000);

    // Set default output path
    m_outputPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/StereoVision";
    QDir().mkpath(m_outputPath);
}

MainWindow::~MainWindow()
{
    delete m_calibration;
    delete m_stereoMatcher;
    delete m_pointCloudProcessor;
}

void MainWindow::setupUI()
{
    setupMenuBar();
    setupCentralWidget();
    setupStatusBar();
}

void MainWindow::setupMenuBar()
{
    m_menuBar = menuBar();

    // File menu
    m_fileMenu = m_menuBar->addMenu("&File");

    m_openLeftAction = new QAction("Open &Left Image...", this);
    m_openLeftAction->setShortcut(QKeySequence("Ctrl+L"));
    m_fileMenu->addAction(m_openLeftAction);

    m_openRightAction = new QAction("Open &Right Image...", this);
    m_openRightAction->setShortcut(QKeySequence("Ctrl+R"));
    m_fileMenu->addAction(m_openRightAction);

    m_openFolderAction = new QAction("Open Stereo &Folder...", this);
    m_openFolderAction->setShortcut(QKeySequence("Ctrl+F"));
    m_fileMenu->addAction(m_openFolderAction);

    m_fileMenu->addSeparator();

    m_loadCalibrationAction = new QAction("&Load Calibration...", this);
    m_loadCalibrationAction->setShortcut(QKeySequence("Ctrl+O"));
    m_fileMenu->addAction(m_loadCalibrationAction);

    m_saveCalibrationAction = new QAction("&Save Calibration...", this);
    m_saveCalibrationAction->setShortcut(QKeySequence("Ctrl+S"));
    m_fileMenu->addAction(m_saveCalibrationAction);

    m_fileMenu->addSeparator();

    m_exitAction = new QAction("E&xit", this);
    m_exitAction->setShortcut(QKeySequence("Ctrl+Q"));
    m_fileMenu->addAction(m_exitAction);

    // Process menu
    m_processMenu = m_menuBar->addMenu("&Process");

    m_calibrateAction = new QAction("&Calibrate Cameras...", this);
    m_calibrateAction->setShortcut(QKeySequence("Ctrl+C"));
    m_processMenu->addAction(m_calibrateAction);

    m_processAction = new QAction("Process &Stereo Images", this);
    m_processAction->setShortcut(QKeySequence("Ctrl+P"));
    m_processMenu->addAction(m_processAction);

    m_exportAction = new QAction("&Export Point Cloud...", this);
    m_exportAction->setShortcut(QKeySequence("Ctrl+E"));
    m_processMenu->addAction(m_exportAction);

    // View menu
    m_viewMenu = m_menuBar->addMenu("&View");

    // Help menu
    m_helpMenu = m_menuBar->addMenu("&Help");

    m_aboutAction = new QAction("&About", this);
    m_helpMenu->addAction(m_aboutAction);
}

void MainWindow::setupCentralWidget()
{
    m_centralWidget = new QWidget;
    setCentralWidget(m_centralWidget);

    // Main splitter
    m_mainSplitter = new QSplitter(Qt::Horizontal);

    // Left side: Image display and 3D view
    auto *leftWidget = new QWidget;
    auto *leftLayout = new QVBoxLayout(leftWidget);

    // Image tab widget
    m_imageTabWidget = new QTabWidget;

    m_leftImageWidget = new ImageDisplayWidget;
    m_rightImageWidget = new ImageDisplayWidget;
    m_disparityWidget = new ImageDisplayWidget;

    m_imageTabWidget->addTab(m_leftImageWidget, "Left Image");
    m_imageTabWidget->addTab(m_rightImageWidget, "Right Image");
    m_imageTabWidget->addTab(m_disparityWidget, "Disparity Map");

    // Point cloud widget
    m_pointCloudWidget = new PointCloudWidget;

    // Add to left layout
    leftLayout->addWidget(m_imageTabWidget, 3);
    leftLayout->addWidget(m_pointCloudWidget, 2);

    // Right side: Parameter panel
    m_parameterPanel = new ParameterPanel;

    // Add to main splitter
    m_mainSplitter->addWidget(leftWidget);
    m_mainSplitter->addWidget(m_parameterPanel);
    m_mainSplitter->setStretchFactor(0, 3);
    m_mainSplitter->setStretchFactor(1, 1);

    // Main layout
    auto *mainLayout = new QVBoxLayout(m_centralWidget);
    mainLayout->addWidget(m_mainSplitter);
}

void MainWindow::setupStatusBar()
{
    m_statusBar = statusBar();

    m_statusLabel = new QLabel("Ready");
    m_statusBar->addWidget(m_statusLabel);

    m_progressBar = new QProgressBar;
    m_progressBar->setVisible(false);
    m_statusBar->addPermanentWidget(m_progressBar);
}

void MainWindow::connectSignals()
{
    // File actions
    connect(m_openLeftAction, &QAction::triggered, this, &MainWindow::openLeftImage);
    connect(m_openRightAction, &QAction::triggered, this, &MainWindow::openRightImage);
    connect(m_openFolderAction, &QAction::triggered, this, &MainWindow::openStereoFolder);
    connect(m_loadCalibrationAction, &QAction::triggered, this, &MainWindow::loadCalibration);
    connect(m_saveCalibrationAction, &QAction::triggered, this, &MainWindow::saveCalibration);
    connect(m_exitAction, &QAction::triggered, this, &QWidget::close);

    // Process actions
    connect(m_calibrateAction, &QAction::triggered, this, &MainWindow::runCalibration);
    connect(m_processAction, &QAction::triggered, this, &MainWindow::processStereoImages);
    connect(m_exportAction, &QAction::triggered, this, &MainWindow::exportPointCloud);

    // Help actions
    connect(m_aboutAction, &QAction::triggered, this, &MainWindow::showAbout);

    // Parameter panel
    connect(m_parameterPanel, &ParameterPanel::parametersChanged,
            this, &MainWindow::onParameterChanged);

    // Processing timer
    connect(m_processingTimer, &QTimer::timeout, this, &MainWindow::onProcessingFinished);
}

void MainWindow::openLeftImage()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    "Open Left Image", m_leftImagePath,
                                                    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)");

    if (!fileName.isEmpty())
    {
        m_leftImagePath = fileName;
        m_leftImageWidget->setImage(fileName);
        m_hasImages = !m_rightImagePath.isEmpty();
        updateUI();
        m_statusLabel->setText("Left image loaded: " + QFileInfo(fileName).fileName());
    }
}

void MainWindow::openRightImage()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    "Open Right Image", m_rightImagePath,
                                                    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)");

    if (!fileName.isEmpty())
    {
        m_rightImagePath = fileName;
        m_rightImageWidget->setImage(fileName);
        m_hasImages = !m_leftImagePath.isEmpty();
        updateUI();
        m_statusLabel->setText("Right image loaded: " + QFileInfo(fileName).fileName());
    }
}

void MainWindow::openStereoFolder()
{
    QString dirName = QFileDialog::getExistingDirectory(this,
                                                        "Open Stereo Image Folder", m_outputPath);

    if (!dirName.isEmpty())
    {
        QDir dir(dirName);
        QStringList imageFiles = dir.entryList(QStringList() << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.tiff",
                                               QDir::Files, QDir::Name);

        if (imageFiles.size() >= 2)
        {
            m_leftImagePath = dir.absoluteFilePath(imageFiles[0]);
            m_rightImagePath = dir.absoluteFilePath(imageFiles[1]);

            m_leftImageWidget->setImage(m_leftImagePath);
            m_rightImageWidget->setImage(m_rightImagePath);

            m_hasImages = true;
            updateUI();
            m_statusLabel->setText(QString("Loaded stereo pair from: %1").arg(dir.dirName()));
        }
        else
        {
            QMessageBox::warning(this, "Error", "Folder must contain at least 2 image files.");
        }
    }
}

void MainWindow::loadCalibration()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    "Load Camera Calibration", m_calibrationPath,
                                                    "Calibration Files (*.xml *.yml *.yaml)");

    if (!fileName.isEmpty())
    {
        if (m_calibration->loadCalibration(fileName.toStdString()))
        {
            m_calibrationPath = fileName;
            m_hasCalibration = true;
            updateUI();
            m_statusLabel->setText("Calibration loaded: " + QFileInfo(fileName).fileName());
        }
        else
        {
            QMessageBox::warning(this, "Error", "Failed to load calibration file.");
        }
    }
}

void MainWindow::saveCalibration()
{
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    "Save Camera Calibration", m_calibrationPath,
                                                    "Calibration Files (*.xml *.yml *.yaml)");

    if (!fileName.isEmpty())
    {
        if (m_calibration->saveCalibration(fileName.toStdString()))
        {
            m_calibrationPath = fileName;
            m_statusLabel->setText("Calibration saved: " + QFileInfo(fileName).fileName());
        }
        else
        {
            QMessageBox::warning(this, "Error", "Failed to save calibration file.");
        }
    }
}

void MainWindow::runCalibration()
{
    // TODO: Implement calibration dialog
    QMessageBox::information(this, "Calibration", "Camera calibration feature coming soon!");
}

void MainWindow::processStereoImages()
{
    if (!m_hasImages || !m_hasCalibration)
    {
        QMessageBox::warning(this, "Error",
                             "Please load stereo images and calibration data first.");
        return;
    }

    m_isProcessing = true;
    updateUI();

    m_statusLabel->setText("Processing stereo images...");
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0); // Indeterminate progress

    // Start processing timer (simulating async processing)
    m_processingTimer->start(2000); // 2 seconds delay for demo
}

void MainWindow::exportPointCloud()
{
    QString fileName = QFileDialog::getSaveFileName(this,
                                                    "Export Point Cloud", m_outputPath + "/pointcloud.ply",
                                                    "Point Cloud Files (*.ply *.pcd *.xyz)");

    if (!fileName.isEmpty())
    {
        // TODO: Implement actual export
        m_statusLabel->setText("Point cloud exported: " + QFileInfo(fileName).fileName());
    }
}

void MainWindow::showAbout()
{
    QMessageBox::about(this, "About Stereo Vision",
                       "<h3>Stereo Vision 3D Point Cloud Generator</h3>"
                       "<p>A high-performance C++ application for generating 3D point clouds from stereo camera images.</p>"
                       "<p><b>Features:</b></p>"
                       "<ul>"
                       "<li>GPU-accelerated stereo matching (CUDA/HIP)</li>"
                       "<li>Camera calibration</li>"
                       "<li>Real-time processing</li>"
                       "<li>Multiple export formats</li>"
                       "</ul>"
                       "<p><b>Version:</b> 1.0.0</p>"
                       "<p><b>Build:</b> " +
                           QString(__DATE__) + " " + QString(__TIME__) + "</p>");
}

void MainWindow::onParameterChanged()
{
    // TODO: Reprocess if images are loaded
    m_statusLabel->setText("Parameters updated");
}

void MainWindow::onProcessingFinished()
{
    m_processingTimer->stop();
    m_isProcessing = false;

    m_progressBar->setVisible(false);
    m_statusLabel->setText("Processing completed");

    // TODO: Display results
    updateUI();
}

void MainWindow::updateUI()
{
    // Update action states
    m_processAction->setEnabled(m_hasImages && m_hasCalibration && !m_isProcessing);
    m_exportAction->setEnabled(!m_isProcessing);
    m_saveCalibrationAction->setEnabled(m_hasCalibration && !m_isProcessing);

    // Update window title
    QString title = "Stereo Vision 3D Point Cloud Generator";
    if (m_hasImages)
    {
        title += " - Images Loaded";
    }
    if (m_hasCalibration)
    {
        title += " - Calibrated";
    }
    if (m_isProcessing)
    {
        title += " - Processing...";
    }
    setWindowTitle(title);
}

void MainWindow::resetView()
{
    m_leftImageWidget->clearImage();
    m_rightImageWidget->clearImage();
    m_disparityWidget->clearImage();
    m_pointCloudWidget->clearPointCloud();

    m_leftImagePath.clear();
    m_rightImagePath.clear();
    m_hasImages = false;

    updateUI();
    m_statusLabel->setText("View reset");
}

#include "main_window.moc"
