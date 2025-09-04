#pragma once

#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>
#include <QStandardPaths>
#include <QString>
#include <QWidget>
#include <memory>

namespace stereo_vision::gui {

/**
 * @brief Enhanced settings manager with user profiles and backup
 */
class SettingsManager {
public:
    static SettingsManager& instance();

    // Profile management
    bool createProfile(const QString& profileName);
    bool switchProfile(const QString& profileName);
    bool deleteProfile(const QString& profileName);
    QStringList getProfiles() const;
    QString getCurrentProfile() const;

    // Settings operations
    void setValue(const QString& key, const QVariant& value);
    QVariant getValue(const QString& key, const QVariant& defaultValue = QVariant()) const;
    bool hasKey(const QString& key) const;
    void removeKey(const QString& key);

    // Backup and restore
    bool exportSettings(const QString& filePath) const;
    bool importSettings(const QString& filePath);
    bool createBackup() const;
    QStringList getBackups() const;
    bool restoreFromBackup(const QString& backupName);

    // Configuration groups
    void beginGroup(const QString& group);
    void endGroup();

    // First-run detection
    bool isFirstRun() const;
    void markAsConfigured();

    // Validation and migration
    bool validateSettings() const;
    bool migrateFromOldVersion();

    // Application settings
    void saveWindowGeometry(QWidget* window, const QString& windowName);
    void restoreWindowGeometry(QWidget* window, const QString& windowName);

private:
    SettingsManager();
    ~SettingsManager() = default;
    SettingsManager(const SettingsManager&) = delete;
    SettingsManager& operator=(const SettingsManager&) = delete;

    QString getProfilePath(const QString& profileName) const;
    QString getBackupPath() const;
    void ensureDirectoriesExist();

    std::unique_ptr<QSettings> m_settings;
    QString m_currentProfile;
    QString m_configDir;
    QString m_profilesDir;
    QString m_backupsDir;
};

/**
 * @brief User profile configuration structure
 */
struct UserProfile {
    QString name;
    QString displayName;
    QString description;
    QString createdDate;
    QString lastUsed;
    bool isDefault = false;

    // User preferences
    QString cameraSetup;     // "single", "dual", "multi"
    QString experienceLevel; // "beginner", "intermediate", "expert"
    QString primaryUseCase;  // "research", "industrial", "hobby"

    QJsonObject toJson() const;
    void fromJson(const QJsonObject& json);
};

/**
 * @brief Configuration template for different use cases
 */
struct ConfigurationTemplate {
    QString name;
    QString description;
    QString category;
    QJsonObject parameters;

    static QList<ConfigurationTemplate> getBuiltInTemplates();
    static ConfigurationTemplate createCustomTemplate(const QString& name,
                                                     const QJsonObject& params);
};

/**
 * @brief Settings validation and migration helper
 */
class SettingsValidator {
public:
    struct ValidationResult {
        bool isValid = true;
        QStringList errors;
        QStringList warnings;
    };

    static ValidationResult validateStereoParameters(const QJsonObject& params);
    static ValidationResult validateCameraSettings(const QJsonObject& settings);
    static ValidationResult validateCalibrationData(const QJsonObject& calibration);

    static QJsonObject migrateFromVersion(const QJsonObject& oldSettings,
                                        const QString& fromVersion,
                                        const QString& toVersion);
};

} // namespace stereo_vision::gui
