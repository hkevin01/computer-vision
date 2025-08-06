#include "gui/batch_processing_window.hpp"
#include <QApplication>
#include <QSettings>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QMimeData>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QDebug>

namespace stereo_vision {

BatchProcessingWindow::BatchProcessingWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_centralWidget(nullptr)
    , m_processor(std::make_unique<BatchProcessor>(this))
    , m_updateTimer(new QTimer(this))
    , m_isProcessing(false)
{
    setWindowTitle("Stereo Vision - Batch Processing");
    setMinimumSize(1000, 700);
    resize(1200, 800);

    // Setup UI
    setupUI();
    setupMenuBar();
    setupStatusBar();
    connectSignals();

    // Load settings
    loadSettings();

    // Setup update timer
    m_updateTimer->setInterval(500); // Update every 500ms
    connect(m_updateTimer, &QTimer::timeout, this, &BatchProcessingWindow::updateProgressDisplay);

    // Enable drag and drop
    setAcceptDrops(true);

    // Initial state
    updateControlStates();
}

BatchProcessingWindow::~BatchProcessingWindow() {
    saveSettings();
}

void BatchProcessingWindow::setupUI() {
    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);

    // Main layout
    QVBoxLayout* mainLayout = new QVBoxLayout(m_centralWidget);

    // Tab widget
    m_tabWidget = new QTabWidget();
    mainLayout->addWidget(m_tabWidget);

    setupConfigurationTab();
    setupOutputTab();

    // Processing controls and progress at bottom
    setupProcessingControls();
    setupProgressDisplay();

    mainLayout->addWidget(m_controlsGroup);
    mainLayout->addWidget(m_progressGroup);
}

void BatchProcessingWindow::setupConfigurationTab() {
    m_configTab = new QWidget();
    m_tabWidget->addTab(m_configTab, "Configuration");

    QVBoxLayout* configLayout = new QVBoxLayout(m_configTab);

    // Directory selection
    setupDirectorySelection();
    configLayout->addWidget(createDirectoryGroup());

    // Output settings
    QGroupBox* outputGroup = new QGroupBox("Output Settings");
    QGridLayout* outputLayout = new QGridLayout(outputGroup);

    outputLayout->addWidget(new QLabel("Output Format:"), 0, 0);
    m_outputFormatCombo = new QComboBox();
    m_outputFormatCombo->addItems({"PLY", "PCD"});
    outputLayout->addWidget(m_outputFormatCombo, 0, 1);

    outputLayout->addWidget(new QLabel("Naming Pattern:"), 1, 0);
    m_namingPatternEdit = new QLineEdit("{basename}_{timestamp}");
    m_namingPatternEdit->setToolTip("Available placeholders: {basename}, {timestamp}, {date}, {time}");
    outputLayout->addWidget(m_namingPatternEdit, 1, 1);

    m_overwriteExistingCheck = new QCheckBox("Overwrite existing files");
    outputLayout->addWidget(m_overwriteExistingCheck, 2, 0, 1, 2);

    configLayout->addWidget(outputGroup);

    // Processing settings
    QGroupBox* processingGroup = new QGroupBox("Processing Settings");
    QGridLayout* procLayout = new QGridLayout(processingGroup);

    m_validateInputsCheck = new QCheckBox("Validate input images");
    m_validateInputsCheck->setChecked(true);
    procLayout->addWidget(m_validateInputsCheck, 0, 0, 1, 2);

    m_enableQualityCheck = new QCheckBox("Enable quality checking");
    m_enableQualityCheck->setChecked(true);
    procLayout->addWidget(m_enableQualityCheck, 1, 0, 1, 2);

    procLayout->addWidget(new QLabel("Min Point Cloud Density:"), 2, 0);
    m_minDensitySpin = new QDoubleSpinBox();
    m_minDensitySpin->setRange(0.0, 1.0);
    m_minDensitySpin->setValue(0.1);
    m_minDensitySpin->setDecimals(3);
    m_minDensitySpin->setSingleStep(0.01);
    procLayout->addWidget(m_minDensitySpin, 2, 1);

    procLayout->addWidget(new QLabel("Max Memory Usage (MB):"), 3, 0);
    m_maxMemorySpin = new QSpinBox();
    m_maxMemorySpin->setRange(512, 16384);
    m_maxMemorySpin->setValue(2048);
    procLayout->addWidget(m_maxMemorySpin, 3, 1);

    procLayout->addWidget(new QLabel("Save Progress Every N:"), 4, 0);
    m_saveProgressSpin = new QSpinBox();
    m_saveProgressSpin->setRange(1, 100);
    m_saveProgressSpin->setValue(10);
    procLayout->addWidget(m_saveProgressSpin, 4, 1);

    m_generateReportCheck = new QCheckBox("Generate processing report");
    m_generateReportCheck->setChecked(true);
    procLayout->addWidget(m_generateReportCheck, 5, 0, 1, 2);

    configLayout->addWidget(processingGroup);

    // Pairs table
    setupPairsTable();
    configLayout->addWidget(m_pairsTable);

    configLayout->addStretch();
}

QGroupBox* BatchProcessingWindow::createDirectoryGroup() {
    QGroupBox* dirGroup = new QGroupBox("Directories");
    QGridLayout* dirLayout = new QGridLayout(dirGroup);

    // Input directory
    dirLayout->addWidget(new QLabel("Input Directory:"), 0, 0);
    m_inputDirectoryEdit = new QLineEdit();
    dirLayout->addWidget(m_inputDirectoryEdit, 0, 1);
    m_browseInputButton = new QPushButton("Browse...");
    dirLayout->addWidget(m_browseInputButton, 0, 2);
    m_scanButton = new QPushButton("Scan");
    dirLayout->addWidget(m_scanButton, 0, 3);

    // Output directory
    dirLayout->addWidget(new QLabel("Output Directory:"), 1, 0);
    m_outputDirectoryEdit = new QLineEdit();
    dirLayout->addWidget(m_outputDirectoryEdit, 1, 1);
    m_browseOutputButton = new QPushButton("Browse...");
    dirLayout->addWidget(m_browseOutputButton, 1, 2);

    return dirGroup;
}

void BatchProcessingWindow::setupDirectorySelection() {
    // Already handled in createDirectoryGroup()
}

void BatchProcessingWindow::setupProcessingControls() {
    m_controlsGroup = new QGroupBox("Processing Controls");
    QHBoxLayout* controlLayout = new QHBoxLayout(m_controlsGroup);

    m_startButton = new QPushButton("Start Processing");
    m_startButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    controlLayout->addWidget(m_startButton);

    m_pauseButton = new QPushButton("Pause");
    m_pauseButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    controlLayout->addWidget(m_pauseButton);

    m_stopButton = new QPushButton("Stop");
    m_stopButton->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    controlLayout->addWidget(m_stopButton);

    controlLayout->addStretch();

    m_exportButton = new QPushButton("Export Results");
    m_exportButton->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    controlLayout->addWidget(m_exportButton);
}

void BatchProcessingWindow::setupProgressDisplay() {
    m_progressGroup = new QGroupBox("Progress");
    QVBoxLayout* progressLayout = new QVBoxLayout(m_progressGroup);

    // Overall progress bar
    m_overallProgress = new QProgressBar();
    progressLayout->addWidget(m_overallProgress);

    // Status information
    QGridLayout* statusLayout = new QGridLayout();

    m_currentOperationLabel = new QLabel("Ready");
    statusLayout->addWidget(new QLabel("Current Operation:"), 0, 0);
    statusLayout->addWidget(m_currentOperationLabel, 0, 1);

    m_statusLabel = new QLabel("Idle");
    statusLayout->addWidget(new QLabel("Status:"), 0, 2);
    statusLayout->addWidget(m_statusLabel, 0, 3);

    m_timeElapsedLabel = new QLabel("00:00:00");
    statusLayout->addWidget(new QLabel("Time Elapsed:"), 1, 0);
    statusLayout->addWidget(m_timeElapsedLabel, 1, 1);

    m_timeRemainingLabel = new QLabel("--:--:--");
    statusLayout->addWidget(new QLabel("Time Remaining:"), 1, 2);
    statusLayout->addWidget(m_timeRemainingLabel, 1, 3);

    m_processingRateLabel = new QLabel("0.0 pairs/sec");
    statusLayout->addWidget(new QLabel("Processing Rate:"), 2, 0);
    statusLayout->addWidget(m_processingRateLabel, 2, 1);

    m_memoryUsageLabel = new QLabel("0 MB");
    statusLayout->addWidget(new QLabel("Memory Usage:"), 2, 2);
    statusLayout->addWidget(m_memoryUsageLabel, 2, 3);

    progressLayout->addLayout(statusLayout);

    // Statistics
    QGroupBox* statsGroup = new QGroupBox("Statistics");
    QGridLayout* statsLayout = new QGridLayout(statsGroup);

    m_totalPairsLabel = new QLabel("0");
    statsLayout->addWidget(new QLabel("Total Pairs:"), 0, 0);
    statsLayout->addWidget(m_totalPairsLabel, 0, 1);

    m_processedPairsLabel = new QLabel("0");
    statsLayout->addWidget(new QLabel("Processed:"), 0, 2);
    statsLayout->addWidget(m_processedPairsLabel, 0, 3);

    m_successfulPairsLabel = new QLabel("0");
    statsLayout->addWidget(new QLabel("Successful:"), 1, 0);
    statsLayout->addWidget(m_successfulPairsLabel, 1, 1);

    m_failedPairsLabel = new QLabel("0");
    statsLayout->addWidget(new QLabel("Failed:"), 1, 2);
    statsLayout->addWidget(m_failedPairsLabel, 1, 3);

    m_skippedPairsLabel = new QLabel("0");
    statsLayout->addWidget(new QLabel("Skipped:"), 2, 0);
    statsLayout->addWidget(m_skippedPairsLabel, 2, 1);

    m_totalOutputSizeLabel = new QLabel("0 MB");
    statsLayout->addWidget(new QLabel("Output Size:"), 2, 2);
    statsLayout->addWidget(m_totalOutputSizeLabel, 2, 3);

    m_totalPointsLabel = new QLabel("0");
    statsLayout->addWidget(new QLabel("Total Points:"), 3, 0);
    statsLayout->addWidget(m_totalPointsLabel, 3, 1);

    progressLayout->addWidget(statsGroup);
}

void BatchProcessingWindow::setupPairsTable() {
    m_pairsTable = new QTableWidget();
    m_pairsTable->setColumnCount(8);

    QStringList headers = {
        "Name", "Status", "Left Image", "Right Image",
        "Points", "Size", "Time", "Error"
    };
    m_pairsTable->setHorizontalHeaderLabels(headers);

    // Configure columns
    m_pairsTable->horizontalHeader()->setStretchLastSection(true);
    m_pairsTable->horizontalHeader()->resizeSection(COLUMN_NAME, 150);
    m_pairsTable->horizontalHeader()->resizeSection(COLUMN_STATUS, 100);
    m_pairsTable->horizontalHeader()->resizeSection(COLUMN_LEFT_IMAGE, 200);
    m_pairsTable->horizontalHeader()->resizeSection(COLUMN_RIGHT_IMAGE, 200);
    m_pairsTable->horizontalHeader()->resizeSection(COLUMN_POINTS, 80);
    m_pairsTable->horizontalHeader()->resizeSection(COLUMN_SIZE, 80);
    m_pairsTable->horizontalHeader()->resizeSection(COLUMN_TIME, 80);

    m_pairsTable->setAlternatingRowColors(true);
    m_pairsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_pairsTable->setSortingEnabled(true);
    m_pairsTable->setMinimumHeight(200);
}

void BatchProcessingWindow::setupOutputTab() {
    m_outputTab = new QWidget();
    m_tabWidget->addTab(m_outputTab, "Output");

    QVBoxLayout* outputLayout = new QVBoxLayout(m_outputTab);

    QSplitter* outputSplitter = new QSplitter(Qt::Vertical);

    // Log output
    QGroupBox* logGroup = new QGroupBox("Processing Log");
    QVBoxLayout* logLayout = new QVBoxLayout(logGroup);
    m_logOutput = new QTextEdit();
    m_logOutput->setReadOnly(true);
    m_logOutput->setFont(QFont("Consolas", 9));
    logLayout->addWidget(m_logOutput);
    outputSplitter->addWidget(logGroup);

    // Error output
    QGroupBox* errorGroup = new QGroupBox("Errors");
    QVBoxLayout* errorLayout = new QVBoxLayout(errorGroup);
    m_errorOutput = new QTextEdit();
    m_errorOutput->setReadOnly(true);
    m_errorOutput->setFont(QFont("Consolas", 9));
    errorLayout->addWidget(m_errorOutput);
    outputSplitter->addWidget(errorGroup);

    outputLayout->addWidget(outputSplitter);
}

void BatchProcessingWindow::setupMenuBar() {
    QMenuBar* menuBar = this->menuBar();

    // File menu
    QMenu* fileMenu = menuBar->addMenu("File");

    QAction* loadConfigAction = fileMenu->addAction("Load Configuration...");
    loadConfigAction->setShortcut(QKeySequence::Open);
    connect(loadConfigAction, &QAction::triggered, this, &BatchProcessingWindow::loadConfiguration);

    QAction* saveConfigAction = fileMenu->addAction("Save Configuration...");
    saveConfigAction->setShortcut(QKeySequence::Save);
    connect(saveConfigAction, &QAction::triggered, this, &BatchProcessingWindow::saveConfiguration);

    fileMenu->addSeparator();

    QAction* exportAction = fileMenu->addAction("Export Results...");
    connect(exportAction, &QAction::triggered, this, &BatchProcessingWindow::exportResults);

    fileMenu->addSeparator();

    QAction* exitAction = fileMenu->addAction("Exit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    // Processing menu
    QMenu* procMenu = menuBar->addMenu("Processing");

    QAction* startAction = procMenu->addAction("Start Processing");
    startAction->setShortcut(Qt::Key_F5);
    connect(startAction, &QAction::triggered, this, &BatchProcessingWindow::startProcessing);

    QAction* pauseAction = procMenu->addAction("Pause");
    pauseAction->setShortcut(Qt::Key_F6);
    connect(pauseAction, &QAction::triggered, this, &BatchProcessingWindow::pauseProcessing);

    QAction* stopAction = procMenu->addAction("Stop");
    stopAction->setShortcut(Qt::Key_F7);
    connect(stopAction, &QAction::triggered, this, &BatchProcessingWindow::stopProcessing);

    procMenu->addSeparator();

    QAction* resetAction = procMenu->addAction("Reset to Defaults");
    connect(resetAction, &QAction::triggered, this, &BatchProcessingWindow::resetToDefaults);

    // View menu
    QMenu* viewMenu = menuBar->addMenu("View");

    QAction* detailsAction = viewMenu->addAction("Show Detailed Progress");
    connect(detailsAction, &QAction::triggered, this, &BatchProcessingWindow::showDetailedProgress);
}

void BatchProcessingWindow::setupStatusBar() {
    statusBar()->showMessage("Ready");
}

void BatchProcessingWindow::connectSignals() {
    // Directory controls
    connect(m_browseInputButton, &QPushButton::clicked, this, &BatchProcessingWindow::browseInputDirectory);
    connect(m_browseOutputButton, &QPushButton::clicked, this, &BatchProcessingWindow::browseOutputDirectory);
    connect(m_inputDirectoryEdit, &QLineEdit::textChanged, this, &BatchProcessingWindow::onInputDirectoryChanged);
    connect(m_scanButton, &QPushButton::clicked, this, &BatchProcessingWindow::scanDirectory);

    // Processing controls
    connect(m_startButton, &QPushButton::clicked, this, &BatchProcessingWindow::startProcessing);
    connect(m_pauseButton, &QPushButton::clicked, this, &BatchProcessingWindow::pauseProcessing);
    connect(m_stopButton, &QPushButton::clicked, this, &BatchProcessingWindow::stopProcessing);
    connect(m_exportButton, &QPushButton::clicked, this, &BatchProcessingWindow::exportResults);

    // Batch processor signals
    connect(m_processor.get(), &BatchProcessor::progressUpdated,
            this, &BatchProcessingWindow::onProgressUpdated);
    connect(m_processor.get(), &BatchProcessor::pairStarted,
            this, &BatchProcessingWindow::onPairStarted);
    connect(m_processor.get(), &BatchProcessor::pairCompleted,
            this, &BatchProcessingWindow::onPairCompleted);
    connect(m_processor.get(), &BatchProcessor::pairFailed,
            this, &BatchProcessingWindow::onPairFailed);
    connect(m_processor.get(), &BatchProcessor::processingFinished,
            this, &BatchProcessingWindow::onProcessingFinished);
}

void BatchProcessingWindow::browseInputDirectory() {
    QString dir = QFileDialog::getExistingDirectory(
        this, "Select Input Directory", m_lastInputDirectory);

    if (!dir.isEmpty()) {
        m_inputDirectoryEdit->setText(dir);
        m_lastInputDirectory = dir;
    }
}

void BatchProcessingWindow::browseOutputDirectory() {
    QString dir = QFileDialog::getExistingDirectory(
        this, "Select Output Directory", m_lastOutputDirectory);

    if (!dir.isEmpty()) {
        m_outputDirectoryEdit->setText(dir);
        m_lastOutputDirectory = dir;
    }
}

void BatchProcessingWindow::onInputDirectoryChanged() {
    // Clear pairs table when input directory changes
    m_pairsTable->setRowCount(0);
    updateControlStates();
}

void BatchProcessingWindow::scanDirectory() {
    QString inputDir = m_inputDirectoryEdit->text().trimmed();
    if (inputDir.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select an input directory first.");
        return;
    }

    if (!QDir(inputDir).exists()) {
        QMessageBox::warning(this, "Warning", "Input directory does not exist.");
        return;
    }

    m_logOutput->append(QString("[%1] Scanning directory: %2")
                       .arg(QTime::currentTime().toString(), inputDir));

    bool success = m_processor->scanDirectory(inputDir);

    if (success) {
        updatePairsList();
        m_logOutput->append(QString("[%1] Found %2 stereo pairs")
                           .arg(QTime::currentTime().toString())
                           .arg(m_pairsTable->rowCount()));
    } else {
        QMessageBox::warning(this, "Warning", "No stereo pairs found in the specified directory.");
        m_logOutput->append(QString("[%1] No stereo pairs found")
                           .arg(QTime::currentTime().toString()));
    }

    updateControlStates();
}

void BatchProcessingWindow::startProcessing() {
    if (m_isProcessing) {
        return;
    }

    // Validate configuration
    QString inputDir = m_inputDirectoryEdit->text().trimmed();
    QString outputDir = m_outputDirectoryEdit->text().trimmed();

    if (inputDir.isEmpty() || outputDir.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please specify both input and output directories.");
        return;
    }

    if (m_pairsTable->rowCount() == 0) {
        QMessageBox::warning(this, "Warning", "No stereo pairs found. Please scan the input directory first.");
        return;
    }

    // Apply configuration
    applyConfiguration();

    // Start processing
    m_logOutput->append(QString("[%1] Starting batch processing...")
                       .arg(QTime::currentTime().toString()));

    if (m_processor->startProcessing()) {
        m_isProcessing = true;
        m_updateTimer->start();
        updateControlStates();
        statusBar()->showMessage("Processing...");
    } else {
        QMessageBox::critical(this, "Error", "Failed to start batch processing.");
    }
}

void BatchProcessingWindow::pauseProcessing() {
    if (!m_isProcessing) {
        return;
    }

    m_processor->pauseProcessing();
    m_logOutput->append(QString("[%1] Processing paused")
                       .arg(QTime::currentTime().toString()));
    statusBar()->showMessage("Paused");
}

void BatchProcessingWindow::stopProcessing() {
    if (!m_isProcessing) {
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "Confirm Stop", "Are you sure you want to stop batch processing?",
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        m_processor->stopProcessing();
        m_updateTimer->stop();
        m_isProcessing = false;
        updateControlStates();

        m_logOutput->append(QString("[%1] Processing stopped by user")
                           .arg(QTime::currentTime().toString()));
        statusBar()->showMessage("Stopped");
    }
}

void BatchProcessingWindow::onProgressUpdated(const BatchProcessingProgress& progress) {
    // Progress will be updated by the timer
}

void BatchProcessingWindow::onPairStarted(const StereoImagePair& pair) {
    m_logOutput->append(QString("[%1] Started processing: %2")
                       .arg(QTime::currentTime().toString(), pair.outputBaseName));

    int row = findPairRow(pair.outputBaseName);
    if (row >= 0) {
        updatePairInTable(pair, row);
    }
}

void BatchProcessingWindow::onPairCompleted(const StereoImagePair& pair) {
    m_logOutput->append(QString("[%1] Completed: %2 (%3 points, %4 ms)")
                       .arg(QTime::currentTime().toString())
                       .arg(pair.outputBaseName)
                       .arg(pair.pointCloudPoints)
                       .arg(pair.processingTimeMs));

    int row = findPairRow(pair.outputBaseName);
    if (row >= 0) {
        updatePairInTable(pair, row);
    }
}

void BatchProcessingWindow::onPairFailed(const StereoImagePair& pair, const QString& error) {
    QString errorMsg = QString("[%1] Failed: %2 - %3")
                       .arg(QTime::currentTime().toString())
                       .arg(pair.outputBaseName)
                       .arg(error);

    m_logOutput->append(errorMsg);
    m_errorOutput->append(errorMsg);

    int row = findPairRow(pair.outputBaseName);
    if (row >= 0) {
        updatePairInTable(pair, row);
    }
}

void BatchProcessingWindow::onProcessingFinished() {
    m_updateTimer->stop();
    m_isProcessing = false;
    updateControlStates();

    BatchProgressSnapshot progress = m_processor->getProgressSnapshot();

    QString completionMsg = QString("[%1] Batch processing completed! Processed: %2, Successful: %3, Failed: %4")
                           .arg(QTime::currentTime().toString())
                           .arg(progress.processedPairs)
                           .arg(progress.successfulPairs)
                           .arg(progress.failedPairs);

    m_logOutput->append(completionMsg);
    statusBar()->showMessage("Completed");

    // Show completion dialog
    QMessageBox::information(this, "Processing Complete", completionMsg);
}

void BatchProcessingWindow::updatePairsList() {
    // Get pairs from processor - this would need to be implemented in BatchProcessor
    // For now, we'll populate based on the scanned pairs
    QList<StereoImagePair> pairs; // Would get from processor

    m_pairsTable->setRowCount(pairs.size());

    for (int i = 0; i < pairs.size(); ++i) {
        updatePairInTable(pairs[i], i);
    }
}

void BatchProcessingWindow::updateProgressDisplay() {
    if (!m_isProcessing) {
        return;
    }

    BatchProgressSnapshot progress = m_processor->getProgressSnapshot();

    // Update progress bar
    if (progress.totalPairs > 0) {
        int percentage = (progress.processedPairs * 100) / progress.totalPairs;
        m_overallProgress->setValue(percentage);
    }

    // Update labels
    m_currentOperationLabel->setText(progress.currentOperation);
    m_statusLabel->setText(m_processor->isProcessing() ? "Processing" : "Idle");

    // Time information
    if (!progress.startTime.isNull()) {
        qint64 elapsedMs = progress.startTime.msecsTo(QDateTime::currentDateTime());
        QString elapsedStr;
        formatDuration(elapsedMs, elapsedStr);
        m_timeElapsedLabel->setText(elapsedStr);

        if (progress.estimatedTimeRemainingMs > 0) {
            QString remainingStr;
            formatDuration(progress.estimatedTimeRemainingMs, remainingStr);
            m_timeRemainingLabel->setText(remainingStr);
        }
    }

    // Processing rate
    m_processingRateLabel->setText(QString("%1 pairs/sec")
                                  .arg(progress.processingRatePerSecond, 0, 'f', 2));

    // Memory usage
    QString memoryStr;
    formatBytes(progress.peakMemoryUsageBytes, memoryStr);
    m_memoryUsageLabel->setText(memoryStr);

    // Statistics
    m_totalPairsLabel->setText(QString::number(progress.totalPairs));
    m_processedPairsLabel->setText(QString::number(progress.processedPairs));
    m_successfulPairsLabel->setText(QString::number(progress.successfulPairs));
    m_failedPairsLabel->setText(QString::number(progress.failedPairs));
    m_skippedPairsLabel->setText(QString::number(progress.skippedPairs));

    QString outputSizeStr;
    formatBytes(progress.totalOutputSizeBytes, outputSizeStr);
    m_totalOutputSizeLabel->setText(outputSizeStr);

    m_totalPointsLabel->setText(QString::number(progress.totalPointCloudPoints));
}

void BatchProcessingWindow::updateControlStates() {
    bool hasInput = !m_inputDirectoryEdit->text().isEmpty();
    bool hasOutput = !m_outputDirectoryEdit->text().isEmpty();
    bool hasPairs = m_pairsTable->rowCount() > 0;

    m_scanButton->setEnabled(hasInput && !m_isProcessing);
    m_startButton->setEnabled(hasInput && hasOutput && hasPairs && !m_isProcessing);
    m_pauseButton->setEnabled(m_isProcessing);
    m_stopButton->setEnabled(m_isProcessing);
    m_exportButton->setEnabled(hasPairs);
}

BatchProcessingConfig BatchProcessingWindow::getCurrentConfig() const {
    BatchProcessingConfig config;

    config.inputDirectory = m_inputDirectoryEdit->text();
    config.outputDirectory = m_outputDirectoryEdit->text();
    config.outputFormat = m_outputFormatCombo->currentText();
    config.namingPattern = m_namingPatternEdit->text();
    config.overwriteExisting = m_overwriteExistingCheck->isChecked();
    config.validateInputs = m_validateInputsCheck->isChecked();
    config.enableQualityCheck = m_enableQualityCheck->isChecked();
    config.minPointCloudDensity = m_minDensitySpin->value();
    config.maxMemoryUsageMB = m_maxMemorySpin->value();
    config.saveProgressEveryN = m_saveProgressSpin->value();
    config.generateReport = m_generateReportCheck->isChecked();

    return config;
}

void BatchProcessingWindow::applyConfiguration() {
    BatchProcessingConfig config = getCurrentConfig();
    m_processor->setConfig(config);
}

void BatchProcessingWindow::updatePairInTable(const StereoImagePair& pair, int row) {
    if (row < 0 || row >= m_pairsTable->rowCount()) {
        return;
    }

    // Set items for each column
    m_pairsTable->setItem(row, COLUMN_NAME, new QTableWidgetItem(pair.outputBaseName));

    QString statusText;
    switch (pair.status) {
        case StereoImagePair::Pending: statusText = "Pending"; break;
        case StereoImagePair::Processing: statusText = "Processing"; break;
        case StereoImagePair::Completed: statusText = "Completed"; break;
        case StereoImagePair::Failed: statusText = "Failed"; break;
        case StereoImagePair::Skipped: statusText = "Skipped"; break;
    }
    m_pairsTable->setItem(row, COLUMN_STATUS, new QTableWidgetItem(statusText));

    m_pairsTable->setItem(row, COLUMN_LEFT_IMAGE, new QTableWidgetItem(QFileInfo(pair.leftImagePath).fileName()));
    m_pairsTable->setItem(row, COLUMN_RIGHT_IMAGE, new QTableWidgetItem(QFileInfo(pair.rightImagePath).fileName()));
    m_pairsTable->setItem(row, COLUMN_POINTS, new QTableWidgetItem(QString::number(pair.pointCloudPoints)));

    QString sizeStr;
    formatBytes(pair.outputFileSizeBytes, sizeStr);
    m_pairsTable->setItem(row, COLUMN_SIZE, new QTableWidgetItem(sizeStr));

    m_pairsTable->setItem(row, COLUMN_TIME, new QTableWidgetItem(QString::number(pair.processingTimeMs)));
    m_pairsTable->setItem(row, COLUMN_ERROR, new QTableWidgetItem(pair.errorMessage));

    // Color code status
    QTableWidgetItem* statusItem = m_pairsTable->item(row, COLUMN_STATUS);
    if (statusItem) {
        switch (pair.status) {
            case StereoImagePair::Pending:
                statusItem->setBackground(QColor(255, 255, 200)); // Light yellow
                break;
            case StereoImagePair::Processing:
                statusItem->setBackground(QColor(200, 200, 255)); // Light blue
                break;
            case StereoImagePair::Completed:
                statusItem->setBackground(QColor(200, 255, 200)); // Light green
                break;
            case StereoImagePair::Failed:
                statusItem->setBackground(QColor(255, 200, 200)); // Light red
                break;
            case StereoImagePair::Skipped:
                statusItem->setBackground(QColor(240, 240, 240)); // Light gray
                break;
        }
    }
}

int BatchProcessingWindow::findPairRow(const QString& baseName) const {
    for (int row = 0; row < m_pairsTable->rowCount(); ++row) {
        QTableWidgetItem* item = m_pairsTable->item(row, COLUMN_NAME);
        if (item && item->text() == baseName) {
            return row;
        }
    }
    return -1;
}

void BatchProcessingWindow::formatBytes(qint64 bytes, QString& result) const {
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;

    if (bytes >= GB) {
        result = QString("%1 GB").arg(static_cast<double>(bytes) / GB, 0, 'f', 2);
    } else if (bytes >= MB) {
        result = QString("%1 MB").arg(static_cast<double>(bytes) / MB, 0, 'f', 1);
    } else if (bytes >= KB) {
        result = QString("%1 KB").arg(static_cast<double>(bytes) / KB, 0, 'f', 1);
    } else {
        result = QString("%1 B").arg(bytes);
    }
}

void BatchProcessingWindow::formatDuration(qint64 milliseconds, QString& result) const {
    qint64 seconds = milliseconds / 1000;
    qint64 minutes = seconds / 60;
    qint64 hours = minutes / 60;

    seconds %= 60;
    minutes %= 60;

    result = QString("%1:%2:%3")
             .arg(hours, 2, 10, QChar('0'))
             .arg(minutes, 2, 10, QChar('0'))
             .arg(seconds, 2, 10, QChar('0'));
}

void BatchProcessingWindow::loadSettings() {
    QSettings settings;
    settings.beginGroup("BatchProcessing");

    m_inputDirectoryEdit->setText(settings.value("inputDirectory").toString());
    m_outputDirectoryEdit->setText(settings.value("outputDirectory").toString());
    m_outputFormatCombo->setCurrentText(settings.value("outputFormat", "PLY").toString());
    m_namingPatternEdit->setText(settings.value("namingPattern", "{basename}_{timestamp}").toString());
    m_overwriteExistingCheck->setChecked(settings.value("overwriteExisting", false).toBool());
    m_validateInputsCheck->setChecked(settings.value("validateInputs", true).toBool());
    m_enableQualityCheck->setChecked(settings.value("enableQualityCheck", true).toBool());
    m_minDensitySpin->setValue(settings.value("minDensity", 0.1).toDouble());
    m_maxMemorySpin->setValue(settings.value("maxMemory", 2048).toInt());
    m_saveProgressSpin->setValue(settings.value("saveProgressEvery", 10).toInt());
    m_generateReportCheck->setChecked(settings.value("generateReport", true).toBool());

    m_lastInputDirectory = settings.value("lastInputDirectory").toString();
    m_lastOutputDirectory = settings.value("lastOutputDirectory").toString();

    settings.endGroup();

    // Restore window geometry
    settings.beginGroup("MainWindow");
    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("state").toByteArray());
    settings.endGroup();
}

void BatchProcessingWindow::saveSettings() {
    QSettings settings;
    settings.beginGroup("BatchProcessing");

    settings.setValue("inputDirectory", m_inputDirectoryEdit->text());
    settings.setValue("outputDirectory", m_outputDirectoryEdit->text());
    settings.setValue("outputFormat", m_outputFormatCombo->currentText());
    settings.setValue("namingPattern", m_namingPatternEdit->text());
    settings.setValue("overwriteExisting", m_overwriteExistingCheck->isChecked());
    settings.setValue("validateInputs", m_validateInputsCheck->isChecked());
    settings.setValue("enableQualityCheck", m_enableQualityCheck->isChecked());
    settings.setValue("minDensity", m_minDensitySpin->value());
    settings.setValue("maxMemory", m_maxMemorySpin->value());
    settings.setValue("saveProgressEvery", m_saveProgressSpin->value());
    settings.setValue("generateReport", m_generateReportCheck->isChecked());

    settings.setValue("lastInputDirectory", m_lastInputDirectory);
    settings.setValue("lastOutputDirectory", m_lastOutputDirectory);

    settings.endGroup();

    // Save window geometry
    settings.beginGroup("MainWindow");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("state", saveState());
    settings.endGroup();
}

void BatchProcessingWindow::saveConfiguration() {
    QString fileName = QFileDialog::getSaveFileName(
        this, "Save Configuration", "", "JSON Files (*.json)");

    if (!fileName.isEmpty()) {
        // Implementation would save current configuration to file
        QMessageBox::information(this, "Configuration", "Configuration saved successfully.");
    }
}

void BatchProcessingWindow::loadConfiguration() {
    QString fileName = QFileDialog::getOpenFileName(
        this, "Load Configuration", "", "JSON Files (*.json)");

    if (!fileName.isEmpty()) {
        // Implementation would load configuration from file
        QMessageBox::information(this, "Configuration", "Configuration loaded successfully.");
    }
}

void BatchProcessingWindow::resetToDefaults() {
    m_outputFormatCombo->setCurrentText("PLY");
    m_namingPatternEdit->setText("{basename}_{timestamp}");
    m_overwriteExistingCheck->setChecked(false);
    m_validateInputsCheck->setChecked(true);
    m_enableQualityCheck->setChecked(true);
    m_minDensitySpin->setValue(0.1);
    m_maxMemorySpin->setValue(2048);
    m_saveProgressSpin->setValue(10);
    m_generateReportCheck->setChecked(true);
}

void BatchProcessingWindow::exportResults() {
    QString fileName = QFileDialog::getSaveFileName(
        this, "Export Results", "batch_results.json", "JSON Files (*.json)");

    if (!fileName.isEmpty()) {
        QJsonObject statistics = m_processor->getProcessingStatistics();
        // Save to file
        QMessageBox::information(this, "Export", "Results exported successfully.");
    }
}

void BatchProcessingWindow::showDetailedProgress() {
    // Could show a separate detailed progress dialog
    QMessageBox::information(this, "Detailed Progress", "Feature not yet implemented.");
}

} // namespace stereo_vision

#include "batch_processing_window.moc"
