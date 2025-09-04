#pragma once

#include "parameter_panel.hpp"
#include "settings_manager.hpp"
#include <QComboBox>
#include <QPushButton>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QTimer>
#include <QProgressBar>
#include <QTextEdit>
#include <QDialog>
#include <QLineEdit>
#include <QCheckBox>

namespace stereo_vision::gui {

/**
 * @brief Enhanced parameter panel with profile management and real-time validation
 */
class EnhancedParameterPanel : public ParameterPanel {
    Q_OBJECT

public:
    explicit EnhancedParameterPanel(QWidget* parent = nullptr);
    ~EnhancedParameterPanel() = default;

    // Profile management
    void setCurrentProfile(const QString& profileName);
    QString getCurrentProfile() const;

    // Parameter validation
    bool validateCurrentParameters() const;
    QStringList getValidationErrors() const;
    QStringList getValidationWarnings() const;

    // Real-time updates
    void setRealTimeValidation(bool enabled);
    bool isRealTimeValidationEnabled() const;

public slots:
    void refreshFromProfile();
    void saveToProfile();
    void resetToDefaults();
    void applyTemplate(const QString& templateName);

    // Profile operations
    void createNewProfile();
    void deleteCurrentProfile();
    void exportProfile();
    void importProfile();

signals:
    void profileChanged(const QString& profileName);
    void parametersChanged();
    void validationStatusChanged(bool isValid);
    void templateApplied(const QString& templateName);

private slots:
    void onParameterChanged();
    void onProfileSelected();
    void onTemplateSelected();
    void onValidationTimer();
    void showValidationDetails();

private:
    // UI Components
    QComboBox* m_profileCombo;
    QPushButton* m_newProfileButton;
    QPushButton* m_deleteProfileButton;
    QPushButton* m_saveButton;
    QPushButton* m_resetButton;

    QComboBox* m_templateCombo;
    QPushButton* m_applyTemplateButton;

    QLabel* m_validationStatusLabel;
    QPushButton* m_validationDetailsButton;
    QProgressBar* m_validationProgress;

    QPushButton* m_exportButton;
    QPushButton* m_importButton;

    // Validation
    QTimer* m_validationTimer;
    bool m_realTimeValidation;
    mutable QStringList m_lastErrors;
    mutable QStringList m_lastWarnings;

    // Settings
    SettingsManager* m_settingsManager;
    QString m_currentProfile;

    // UI Setup
    void setupProfileControls();
    void setupTemplateControls();
    void setupValidationControls();
    void setupExportImportControls();
    void updateProfileList();
    void updateTemplateList();
    void updateValidationStatus();
    void connectParameterSignals();

    // Validation helpers
    void performValidation() const;
    void showValidationDialog();
    QString formatValidationMessage(const QStringList& errors, const QStringList& warnings) const;

    // Profile helpers
    void loadProfileParameters(const QString& profileName);
    void saveParametersToProfile(const QString& profileName);
    bool confirmProfileDeletion(const QString& profileName);

    // Template helpers
    void applyConfigurationTemplate(const ConfigurationTemplate& config);
    ConfigurationTemplate createTemplateFromCurrentSettings() const;
};

/**
 * @brief Dialog for creating new profiles
 */
class NewProfileDialog : public QDialog {
    Q_OBJECT

public:
    explicit NewProfileDialog(QWidget* parent = nullptr);

    QString getProfileName() const;
    QString getDisplayName() const;
    QString getDescription() const;
    bool shouldCopyFromCurrent() const;

private:
    QLineEdit* m_profileNameEdit;
    QLineEdit* m_displayNameEdit;
    QTextEdit* m_descriptionEdit;
    QCheckBox* m_copyCurrentCheckbox;

    void setupUI();
    bool validateInput() const;

private slots:
    void onAccept();
};

/**
 * @brief Dialog for showing detailed validation results
 */
class ValidationDialog : public QDialog {
    Q_OBJECT

public:
    explicit ValidationDialog(const QStringList& errors,
                            const QStringList& warnings,
                            QWidget* parent = nullptr);

private:
    void setupUI(const QStringList& errors, const QStringList& warnings);
    QString formatErrorsAndWarnings(const QStringList& errors, const QStringList& warnings) const;
};

} // namespace stereo_vision::gui
