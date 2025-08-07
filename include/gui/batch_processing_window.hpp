#ifndef BATCH_PROCESSING_WINDOW_HPP
#define BATCH_PROCESSING_WINDOW_HPP

#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QTextEdit>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QFileDialog>
#include <QTimer>
#include <QHeaderView>
#include <QMessageBox>
#include <QSplitter>
#include <QTabWidget>
#include <QJsonObject>
#include <QStandardPaths>
#include <memory>

#include "batch_processor.hpp"

namespace stereo_vision::batch {

class BatchProcessingWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit BatchProcessingWindow(QWidget* parent = nullptr);
    ~BatchProcessingWindow();

private slots:
    void browseInputDirectory();
    void browseOutputDirectory();
    void onInputDirectoryChanged();
    void scanDirectory();
    void startProcessing();
    void pauseProcessing();
    void stopProcessing();
    void saveConfiguration();
    void loadConfiguration();
    void resetToDefaults();

    // Batch processor signals
    void onProgressUpdated(const BatchProcessingProgress& progress);
    void onPairStarted(const StereoImagePair& pair);
    void onPairCompleted(const StereoImagePair& pair);
    void onPairFailed(const StereoImagePair& pair, const QString& error);
    void onProcessingFinished();

    // UI updates
    void updatePairsList();
    void updateProgressDisplay();
    void updateControlStates();
    void exportResults();
    void showDetailedProgress();

private:
    void setupUI();
    void setupConfigurationTab();
    void setupDirectorySelection();
    void setupProcessingControls();
    void setupProgressDisplay();
    void setupPairsTable();
    void setupOutputTab();
    void setupMenuBar();
    void setupStatusBar();
    QGroupBox* createDirectoryGroup();

    void connectSignals();
    void loadSettings();
    void saveSettings();
    void applyConfiguration();
    BatchProcessingConfig getCurrentConfig() const;
    void updateConfigFromUI();

    void updatePairInTable(const StereoImagePair& pair, int row);
    int findPairRow(const QString& baseName) const;
    void formatBytes(qint64 bytes, QString& result) const;
    void formatDuration(qint64 milliseconds, QString& result) const;

    // UI Components
    QWidget* m_centralWidget;
    QTabWidget* m_tabWidget;
    QSplitter* m_mainSplitter;

    // Configuration Tab
    QWidget* m_configTab;
    QLineEdit* m_inputDirectoryEdit;
    QLineEdit* m_outputDirectoryEdit;
    QPushButton* m_browseInputButton;
    QPushButton* m_browseOutputButton;
    QPushButton* m_scanButton;

    QComboBox* m_outputFormatCombo;
    QLineEdit* m_namingPatternEdit;
    QCheckBox* m_overwriteExistingCheck;
    QCheckBox* m_validateInputsCheck;
    QCheckBox* m_enableQualityCheck;
    QDoubleSpinBox* m_minDensitySpin;
    QSpinBox* m_maxMemorySpin;
    QSpinBox* m_saveProgressSpin;
    QCheckBox* m_generateReportCheck;

    // Processing Controls
    QGroupBox* m_controlsGroup;
    QPushButton* m_startButton;
    QPushButton* m_pauseButton;
    QPushButton* m_stopButton;
    QPushButton* m_exportButton;

    // Progress Display
    QGroupBox* m_progressGroup;
    QProgressBar* m_overallProgress;
    QLabel* m_currentOperationLabel;
    QLabel* m_statusLabel;
    QLabel* m_timeElapsedLabel;
    QLabel* m_timeRemainingLabel;
    QLabel* m_processingRateLabel;
    QLabel* m_memoryUsageLabel;

    // Statistics Display
    QLabel* m_totalPairsLabel;
    QLabel* m_processedPairsLabel;
    QLabel* m_successfulPairsLabel;
    QLabel* m_failedPairsLabel;
    QLabel* m_skippedPairsLabel;
    QLabel* m_totalOutputSizeLabel;
    QLabel* m_totalPointsLabel;

    // Pairs Table
    QTableWidget* m_pairsTable;

    // Output Tab
    QWidget* m_outputTab;
    QTextEdit* m_logOutput;
    QTextEdit* m_errorOutput;

    // Processing
    std::unique_ptr<BatchProcessor> m_processor;
    QTimer* m_updateTimer;

    // State
    bool m_isProcessing;
    QString m_lastInputDirectory;
    QString m_lastOutputDirectory;

    // Constants
    static const int COLUMN_NAME = 0;
    static const int COLUMN_STATUS = 1;
    static const int COLUMN_LEFT_IMAGE = 2;
    static const int COLUMN_RIGHT_IMAGE = 3;
    static const int COLUMN_POINTS = 4;
    static const int COLUMN_SIZE = 5;
    static const int COLUMN_TIME = 6;
    static const int COLUMN_ERROR = 7;
};

} // namespace stereo_vision::batch

#endif // BATCH_PROCESSING_WINDOW_HPP
