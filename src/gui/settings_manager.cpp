#include "gui/settings_manager.hpp"
#include <QApplication>
#include <QDateTime>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLoggingCategory>
#include <QMessageBox>
#include <QDir>
#include <QFile>
#include <QStandardPaths>
#include <QDebug>

Q_LOGGING_CATEGORY(settingsManager, "stereo.gui.settings")

namespace stereo_vision::gui {

SettingsManager& SettingsManager::instance() {
    static SettingsManager instance;
    return instance;
}

SettingsManager::SettingsManager()
    : m_currentProfile("default") {

    // Set up configuration directories
    m_configDir = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    m_profilesDir = m_configDir + "/profiles";
    m_backupsDir = m_configDir + "/backups";

    ensureDirectoriesExist();

    // Initialize with default profile
    if (!switchProfile("default")) {
        createProfile("default");
        switchProfile("default");
    }

    qDebug() << "Settings manager initialized with profile:" << m_currentProfile;
}bool SettingsManager::createProfile(const QString& profileName) {
    if (profileName.isEmpty() || getProfiles().contains(profileName)) {
        qWarning() << "Cannot create profile" << profileName << ": empty name or already exists";
        return false;
    }

    QString profilePath = getProfilePath(profileName);
    QSettings profileSettings(profilePath, QSettings::IniFormat);

    // Initialize with default values
    profileSettings.setValue("profile/name", profileName);
    profileSettings.setValue("profile/created", QDateTime::currentDateTime().toString(Qt::ISODate));
    profileSettings.setValue("profile/version", QApplication::applicationVersion());
    profileSettings.setValue("profile/displayName", profileName);
    profileSettings.setValue("profile/description", QString("Profile created on %1")
                            .arg(QDateTime::currentDateTime().toString()));

    // Set default stereo parameters
    profileSettings.beginGroup("stereo");
    profileSettings.setValue("algorithm", "SGBM");
    profileSettings.setValue("blockSize", 9);
    profileSettings.setValue("minDisparity", 0);
    profileSettings.setValue("numDisparities", 64);
    profileSettings.setValue("enablePostProcessing", true);
    profileSettings.endGroup();

    // Set default camera settings
    profileSettings.beginGroup("camera");
    profileSettings.setValue("leftCameraId", 0);
    profileSettings.setValue("rightCameraId", 1);
    profileSettings.setValue("resolution", "640x480");
    profileSettings.setValue("fps", 30);
    profileSettings.endGroup();

    profileSettings.sync();

    qDebug() << "Created new profile:" << profileName;
    return profileSettings.status() == QSettings::NoError;
}

bool SettingsManager::switchProfile(const QString& profileName) {
    if (profileName.isEmpty()) {
        return false;
    }

    QString profilePath = getProfilePath(profileName);
    if (!QFileInfo::exists(profilePath) && profileName != "default") {
        qWarning() << "Profile" << profileName << "does not exist";
        return false;
    }

    // Save current profile's last used time
    if (m_settings) {
        m_settings->setValue("profile/lastUsed", QDateTime::currentDateTime().toString(Qt::ISODate));
        m_settings->sync();
    }

    // Switch to new profile
    m_settings = std::make_unique<QSettings>(profilePath, QSettings::IniFormat);
    m_currentProfile = profileName;

    // Update last used time for new profile
    m_settings->setValue("profile/lastUsed", QDateTime::currentDateTime().toString(Qt::ISODate));
    m_settings->sync();

    qDebug() << "Switched to profile:" << profileName;
    return true;
}

bool SettingsManager::deleteProfile(const QString& profileName) {
    if (profileName == "default" || profileName == m_currentProfile) {
        qWarning() << "Cannot delete default profile or currently active profile";
        return false;
    }

    QString profilePath = getProfilePath(profileName);
    QFile profileFile(profilePath);

    if (profileFile.exists() && profileFile.remove()) {
        qDebug() << "Deleted profile:" << profileName;
        return true;
    }

    return false;
}

QStringList SettingsManager::getProfiles() const {
    QDir profilesDir(m_profilesDir);
    QStringList profiles;

    QStringList iniFiles = profilesDir.entryList(QStringList("*.ini"), QDir::Files);
    for (const QString& file : iniFiles) {
        QString baseName = QFileInfo(file).baseName();
        profiles.append(baseName);
    }

    if (!profiles.contains("default")) {
        profiles.prepend("default");
    }

    return profiles;
}

QString SettingsManager::getCurrentProfile() const {
    return m_currentProfile;
}

void SettingsManager::setValue(const QString& key, const QVariant& value) {
    if (!m_settings) {
        qCritical() << "Settings not initialized";
        return;
    }

    m_settings->setValue(key, value);
}

QVariant SettingsManager::getValue(const QString& key, const QVariant& defaultValue) const {
    if (!m_settings) {
        qCritical() << "Settings not initialized";
        return defaultValue;
    }

    return m_settings->value(key, defaultValue);
}

bool SettingsManager::hasKey(const QString& key) const {
    if (!m_settings) {
        return false;
    }

    return m_settings->contains(key);
}

void SettingsManager::removeKey(const QString& key) {
    if (!m_settings) {
        return;
    }

    m_settings->remove(key);
}

bool SettingsManager::exportSettings(const QString& filePath) const {
    if (!m_settings) {
        return false;
    }

    try {
        QJsonObject exportObj;

        // Export all settings as JSON
        for (const QString& key : m_settings->allKeys()) {
            QVariant value = m_settings->value(key);

            // Convert QVariant to JSON-compatible format
            if (value.canConvert<QString>()) {
                exportObj[key] = value.toString();
            } else if (value.canConvert<int>()) {
                exportObj[key] = value.toInt();
            } else if (value.canConvert<double>()) {
                exportObj[key] = value.toDouble();
            } else if (value.canConvert<bool>()) {
                exportObj[key] = value.toBool();
            }
        }

        // Add metadata
        QJsonObject metadata;
        metadata["exported"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        metadata["profile"] = m_currentProfile;
        metadata["version"] = QApplication::applicationVersion();
        exportObj["_metadata"] = metadata;

        QJsonDocument doc(exportObj);

        QFile file(filePath);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(doc.toJson());
            qDebug() << "Settings exported to:" << filePath;
            return true;
        }
    } catch (const std::exception& e) {
        qCritical() << "Error exporting settings:" << e.what();
    }

    return false;
}

bool SettingsManager::importSettings(const QString& filePath) {
    if (!m_settings) {
        return false;
    }

    try {
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly)) {
            qCritical() << "Cannot open settings file:" << filePath;
            return false;
        }

        QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        QJsonObject importObj = doc.object();

        // Validate import
        if (!importObj.contains("_metadata")) {
            qWarning() << "Settings file missing metadata, proceeding with caution";
        }

        // Create backup before import
        createBackup();

        // Import settings
        for (auto it = importObj.begin(); it != importObj.end(); ++it) {
            if (it.key().startsWith("_")) {
                continue; // Skip metadata
            }

            QJsonValue value = it.value();
            if (value.isString()) {
                m_settings->setValue(it.key(), value.toString());
            } else if (value.isDouble()) {
                m_settings->setValue(it.key(), value.toDouble());
            } else if (value.isBool()) {
                m_settings->setValue(it.key(), value.toBool());
            }
        }

        m_settings->sync();
        qDebug() << "Settings imported from:" << filePath;
        return true;

    } catch (const std::exception& e) {
        qCritical() << "Error importing settings:" << e.what();
    }

    return false;
}

bool SettingsManager::createBackup() const {
    if (!m_settings) {
        return false;
    }

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString backupFileName = QString("%1_%2.backup").arg(m_currentProfile, timestamp);
    QString backupPath = m_backupsDir + "/" + backupFileName;

    return exportSettings(backupPath);
}

QStringList SettingsManager::getBackups() const {
    QDir backupsDir(m_backupsDir);
    QStringList backups = backupsDir.entryList(QStringList("*.backup"),
                                              QDir::Files,
                                              QDir::Time | QDir::Reversed);
    return backups;
}

bool SettingsManager::restoreFromBackup(const QString& backupName) {
    QString backupPath = m_backupsDir + "/" + backupName;
    return importSettings(backupPath);
}

void SettingsManager::beginGroup(const QString& group) {
    if (m_settings) {
        m_settings->beginGroup(group);
    }
}

void SettingsManager::endGroup() {
    if (m_settings) {
        m_settings->endGroup();
    }
}

bool SettingsManager::isFirstRun() const {
    return !hasKey("app/configured") || !getValue("app/configured", false).toBool();
}

void SettingsManager::markAsConfigured() {
    setValue("app/configured", true);
    setValue("app/configuredDate", QDateTime::currentDateTime().toString(Qt::ISODate));
    setValue("app/configuredVersion", QApplication::applicationVersion());

    if (m_settings) {
        m_settings->sync();
    }
}

bool SettingsManager::validateSettings() const {
    if (!m_settings) {
        return false;
    }

    try {
        // Validate stereo parameters without using const-breaking methods
        QSettings& settings = *m_settings;

        settings.beginGroup("stereo");
        int blockSize = settings.value("blockSize", 9).toInt();
        if (blockSize % 2 == 0 || blockSize < 3 || blockSize > 21) {
            qWarning() << "Invalid block size:" << blockSize;
            settings.endGroup();
            return false;
        }

        int numDisparities = settings.value("numDisparities", 64).toInt();
        if (numDisparities <= 0 || numDisparities % 16 != 0) {
            qWarning() << "Invalid number of disparities:" << numDisparities;
            settings.endGroup();
            return false;
        }
        settings.endGroup();

        // Validate camera settings
        settings.beginGroup("camera");
        int leftId = settings.value("leftCameraId", 0).toInt();
        int rightId = settings.value("rightCameraId", 1).toInt();
        if (leftId == rightId) {
            qWarning() << "Left and right camera IDs cannot be the same";
            settings.endGroup();
            return false;
        }
        settings.endGroup();

        return true;

    } catch (const std::exception& e) {
        qCritical() << "Error validating settings:" << e.what();
        return false;
    }
}

bool SettingsManager::migrateFromOldVersion() {
    // Check if migration is needed
    QString currentVersion = QApplication::applicationVersion();
    QString settingsVersion = getValue("profile/version", "1.0.0").toString();

    if (settingsVersion == currentVersion) {
        return true; // No migration needed
    }

    qDebug() << "Migrating settings from version" << settingsVersion << "to" << currentVersion;    // Create backup before migration
    createBackup();

    // Perform version-specific migrations
    if (settingsVersion.startsWith("1.")) {
        // Migrate from v1.x to v2.x
        // Example: rename old parameter names
        if (hasKey("old_parameter_name")) {
            QVariant value = getValue("old_parameter_name");
            setValue("new_parameter_name", value);
            removeKey("old_parameter_name");
        }
    }

    // Update version
    setValue("profile/version", currentVersion);
    setValue("profile/migrated", QDateTime::currentDateTime().toString(Qt::ISODate));

    if (m_settings) {
        m_settings->sync();
    }

    return true;
}

void SettingsManager::saveWindowGeometry(QWidget* window, const QString& windowName) {
    if (!window || !m_settings) {
        return;
    }

    beginGroup("windows");
    setValue(windowName + "/geometry", window->saveGeometry());
    setValue(windowName + "/state", window->isMaximized());
    endGroup();
}

void SettingsManager::restoreWindowGeometry(QWidget* window, const QString& windowName) {
    if (!window || !m_settings) {
        return;
    }

    beginGroup("windows");
    QByteArray geometry = getValue(windowName + "/geometry").toByteArray();
    bool wasMaximized = getValue(windowName + "/state", false).toBool();
    endGroup();

    if (!geometry.isEmpty()) {
        window->restoreGeometry(geometry);
    }

    if (wasMaximized) {
        window->showMaximized();
    }
}

QString SettingsManager::getProfilePath(const QString& profileName) const {
    return m_profilesDir + "/" + profileName + ".ini";
}

QString SettingsManager::getBackupPath() const {
    return m_backupsDir;
}

void SettingsManager::ensureDirectoriesExist() {
    QDir().mkpath(m_configDir);
    QDir().mkpath(m_profilesDir);
    QDir().mkpath(m_backupsDir);
}

// UserProfile implementation
QJsonObject UserProfile::toJson() const {
    QJsonObject obj;
    obj["name"] = name;
    obj["displayName"] = displayName;
    obj["description"] = description;
    obj["createdDate"] = createdDate;
    obj["lastUsed"] = lastUsed;
    obj["isDefault"] = isDefault;
    obj["cameraSetup"] = cameraSetup;
    obj["experienceLevel"] = experienceLevel;
    obj["primaryUseCase"] = primaryUseCase;
    return obj;
}

void UserProfile::fromJson(const QJsonObject& json) {
    name = json["name"].toString();
    displayName = json["displayName"].toString();
    description = json["description"].toString();
    createdDate = json["createdDate"].toString();
    lastUsed = json["lastUsed"].toString();
    isDefault = json["isDefault"].toBool();
    cameraSetup = json["cameraSetup"].toString();
    experienceLevel = json["experienceLevel"].toString();
    primaryUseCase = json["primaryUseCase"].toString();
}

// ConfigurationTemplate implementation
QList<ConfigurationTemplate> ConfigurationTemplate::getBuiltInTemplates() {
    QList<ConfigurationTemplate> templates;

    // High-quality template
    ConfigurationTemplate highQuality;
    highQuality.name = "High Quality";
    highQuality.description = "Best quality settings for accurate stereo matching";
    highQuality.category = "Quality";
    highQuality.parameters = QJsonObject{
        {"blockSize", 5},
        {"numDisparities", 128},
        {"P1", 8 * 3 * 25},
        {"P2", 32 * 3 * 25},
        {"enableSpeckleFilter", true},
        {"enableMedianFilter", true}
    };
    templates.append(highQuality);

    // Real-time template
    ConfigurationTemplate realTime;
    realTime.name = "Real-time";
    realTime.description = "Fast processing for real-time applications";
    realTime.category = "Performance";
    realTime.parameters = QJsonObject{
        {"blockSize", 9},
        {"numDisparities", 64},
        {"P1", 8 * 9 * 9},
        {"P2", 32 * 9 * 9},
        {"enableSpeckleFilter", false},
        {"enableMedianFilter", false}
    };
    templates.append(realTime);

    // Balanced template
    ConfigurationTemplate balanced;
    balanced.name = "Balanced";
    balanced.description = "Good balance between quality and performance";
    balanced.category = "General";
    balanced.parameters = QJsonObject{
        {"blockSize", 7},
        {"numDisparities", 96},
        {"P1", 8 * 3 * 49},
        {"P2", 32 * 3 * 49},
        {"enableSpeckleFilter", true},
        {"enableMedianFilter", false}
    };
    templates.append(balanced);

    return templates;
}

ConfigurationTemplate ConfigurationTemplate::createCustomTemplate(const QString& name,
                                                                  const QJsonObject& params) {
    ConfigurationTemplate custom;
    custom.name = name;
    custom.description = "Custom user configuration";
    custom.category = "Custom";
    custom.parameters = params;
    return custom;
}

// SettingsValidator implementation
SettingsValidator::ValidationResult SettingsValidator::validateStereoParameters(const QJsonObject& params) {
    ValidationResult result;

    // Validate block size
    int blockSize = params["blockSize"].toInt(9);
    if (blockSize % 2 == 0 || blockSize < 3 || blockSize > 21) {
        result.isValid = false;
        result.errors.append("Block size must be odd and between 3 and 21");
    }

    // Validate number of disparities
    int numDisparities = params["numDisparities"].toInt(64);
    if (numDisparities <= 0 || numDisparities % 16 != 0) {
        result.isValid = false;
        result.errors.append("Number of disparities must be positive and divisible by 16");
    }

    // Validate P1 and P2
    int P1 = params["P1"].toInt();
    int P2 = params["P2"].toInt();
    if (P2 < P1) {
        result.warnings.append("P2 should be larger than P1 for better results");
    }

    return result;
}

SettingsValidator::ValidationResult SettingsValidator::validateCameraSettings(const QJsonObject& settings) {
    ValidationResult result;

    int leftId = settings["leftCameraId"].toInt(0);
    int rightId = settings["rightCameraId"].toInt(1);

    if (leftId == rightId) {
        result.isValid = false;
        result.errors.append("Left and right camera IDs must be different");
    }

    if (leftId < 0 || rightId < 0) {
        result.isValid = false;
        result.errors.append("Camera IDs must be non-negative");
    }

    return result;
}

SettingsValidator::ValidationResult SettingsValidator::validateCalibrationData(const QJsonObject& calibration) {
    ValidationResult result;

    // Check required fields
    QStringList requiredFields = {"camera_matrix_left", "camera_matrix_right",
                                 "dist_coeffs_left", "dist_coeffs_right"};

    for (const QString& field : requiredFields) {
        if (!calibration.contains(field)) {
            result.isValid = false;
            result.errors.append(QString("Missing required calibration field: %1").arg(field));
        }
    }

    return result;
}

QJsonObject SettingsValidator::migrateFromVersion(const QJsonObject& oldSettings,
                                                 const QString& fromVersion,
                                                 const QString& toVersion) {
    QJsonObject newSettings = oldSettings;

    // Version-specific migrations
    if (fromVersion.startsWith("1.") && toVersion.startsWith("2.")) {
        // Example migration: rename parameters
        if (newSettings.contains("old_param")) {
            newSettings["new_param"] = newSettings["old_param"];
            newSettings.remove("old_param");
        }
    }

    return newSettings;
}

} // namespace stereo_vision::gui
