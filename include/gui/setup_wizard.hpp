#pragma once

#include <QWizard>
#include <QWizardPage>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QComboBox>
#include <QSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QTextEdit>
#include <QRadioButton>
#include <QButtonGroup>
#include <QCheckBox>
#include <QGroupBox>
#include <QListWidget>
#include <QSlider>
#include <QFormLayout>

namespace stereo_vision::gui {

// Forward declaration
class SettingsManager;

/**
 * @brief Welcome page for the setup wizard
 */
class WelcomePage : public QWizardPage {
    Q_OBJECT

public:
    WelcomePage(QWidget* parent = nullptr);
    void initializePage() override;
    bool isComplete() const override;

private:
    QLabel* m_welcomeLabel;
    QTextEdit* m_featuresText;
    QCheckBox* m_agreeCheckbox;
};

/**
 * @brief User profile setup page
 */
class ProfileSetupPage : public QWizardPage {
    Q_OBJECT

public:
    ProfileSetupPage(QWidget* parent = nullptr);
    void initializePage() override;
    bool validatePage() override;
    int nextId() const override;

    QString getProfileName() const;
    QString getExperienceLevel() const;
    QString getPrimaryUseCase() const;

private:
    QLineEdit* m_profileNameEdit;
    QLineEdit* m_displayNameEdit;
    QTextEdit* m_descriptionEdit;
    QButtonGroup* m_experienceGroup;
    QButtonGroup* m_useCaseGroup;

    QRadioButton* m_beginnerRadio;
    QRadioButton* m_intermediateRadio;
    QRadioButton* m_advancedRadio;

    QRadioButton* m_researchRadio;
    QRadioButton* m_industrialRadio;
    QRadioButton* m_hobbyRadio;
    QRadioButton* m_educationRadio;
};

/**
 * @brief Camera configuration page
 */
class CameraSetupPage : public QWizardPage {
    Q_OBJECT

public:
    CameraSetupPage(QWidget* parent = nullptr);
    void initializePage() override;
    bool validatePage() override;

    int getLeftCameraId() const;
    int getRightCameraId() const;
    QString getResolution() const;
    int getFps() const;

private slots:
    void detectCameras();
    void testCameraConfiguration();

private:
    QComboBox* m_leftCameraCombo;
    QComboBox* m_rightCameraCombo;
    QComboBox* m_resolutionCombo;
    QSpinBox* m_fpsSpinBox;
    QPushButton* m_detectButton;
    QPushButton* m_testButton;
    QLabel* m_statusLabel;
    QListWidget* m_cameraList;

    void populateResolutions();
    void updateCameraList();
};

/**
 * @brief Stereo algorithm configuration page
 */
class AlgorithmSetupPage : public QWizardPage {
    Q_OBJECT

public:
    AlgorithmSetupPage(QWidget* parent = nullptr);
    void initializePage() override;
    bool validatePage() override;

    QString getAlgorithm() const;
    QVariantMap getStereoParameters() const;

private slots:
    void onAlgorithmChanged();
    void onPresetChanged();
    void resetToDefaults();

private:
    QComboBox* m_algorithmCombo;
    QComboBox* m_presetCombo;
    QSpinBox* m_blockSizeSpinBox;
    QSpinBox* m_numDisparitiesSpinBox;
    QSpinBox* m_minDisparitySpinBox;
    QSlider* m_p1Slider;
    QSlider* m_p2Slider;
    QCheckBox* m_postProcessingCheckbox;
    QCheckBox* m_speckleFilterCheckbox;
    QPushButton* m_resetButton;

    void setupParameterControls();
    void updateParametersForAlgorithm();
    void updateParametersForPreset();
};

/**
 * @brief Calibration setup page
 */
class CalibrationSetupPage : public QWizardPage {
    Q_OBJECT

public:
    CalibrationSetupPage(QWidget* parent = nullptr);
    void initializePage() override;
    bool validatePage() override;
    int nextId() const override;

private slots:
    void startCalibration();
    void loadExistingCalibration();
    void skipCalibration();

private:
    QPushButton* m_startCalibrationButton;
    QPushButton* m_loadCalibrationButton;
    QPushButton* m_skipCalibrationButton;
    QLabel* m_statusLabel;
    QProgressBar* m_progressBar;
    QTextEdit* m_instructionsText;

    bool m_calibrationCompleted;
    bool m_calibrationSkipped;
    QString m_calibrationFile;
};

/**
 * @brief Performance optimization page
 */
class PerformanceSetupPage : public QWizardPage {
    Q_OBJECT

public:
    PerformanceSetupPage(QWidget* parent = nullptr);
    void initializePage() override;
    bool validatePage() override;

    QVariantMap getPerformanceSettings() const;

private slots:
    void runPerformanceBenchmark();
    void onOptimizationModeChanged();

private:
    QButtonGroup* m_optimizationGroup;
    QRadioButton* m_qualityRadio;
    QRadioButton* m_balancedRadio;
    QRadioButton* m_speedRadio;

    QCheckBox* m_useGpuCheckbox;
    QCheckBox* m_multiThreadCheckbox;
    QSpinBox* m_threadCountSpinBox;
    QSlider* m_bufferSizeSlider;

    QPushButton* m_benchmarkButton;
    QProgressBar* m_benchmarkProgress;
    QLabel* m_benchmarkResults;

    void updateThreadControls();
    void updateRecommendations();
};

/**
 * @brief Summary and completion page
 */
class CompletionPage : public QWizardPage {
    Q_OBJECT

public:
    CompletionPage(QWidget* parent = nullptr);
    void initializePage() override;
    bool validatePage() override;

private:
    QTextEdit* m_summaryText;
    QCheckBox* m_createShortcutCheckbox;
    QCheckBox* m_showQuickStartCheckbox;
    QCheckBox* m_enableTelemetryCheckbox;

    void generateSummary();
};

/**
 * @brief Main setup wizard class
 */
class SetupWizard : public QWizard {
    Q_OBJECT

public:
    explicit SetupWizard(QWidget* parent = nullptr);
    ~SetupWizard() = default;

    // Page IDs
    enum PageId {
        Page_Welcome = 0,
        Page_Profile,
        Page_Camera,
        Page_Algorithm,
        Page_Calibration,
        Page_Performance,
        Page_Completion
    };

public slots:
    void accept() override;

private slots:
    void onCurrentIdChanged(int id);

private:
    WelcomePage* m_welcomePage;
    ProfileSetupPage* m_profilePage;
    CameraSetupPage* m_cameraPage;
    AlgorithmSetupPage* m_algorithmPage;
    CalibrationSetupPage* m_calibrationPage;
    PerformanceSetupPage* m_performancePage;
    CompletionPage* m_completionPage;

    SettingsManager* m_settingsManager;

    void setupPages();
    void applySettings();
    void createUserProfile();
    void saveConfiguration();
};

} // namespace stereo_vision::gui
