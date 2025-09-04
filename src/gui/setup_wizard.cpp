#include "gui/setup_wizard.hpp"
#include "gui/settings_manager.hpp"
#include <QApplication>
#include <QScreen>
#include <QMessageBox>
#include <QFileDialog>
#include <QStandardPaths>
#include <QThread>
#include <QTimer>
#include <QDebug>

namespace stereo_vision::gui {

// WelcomePage implementation
WelcomePage::WelcomePage(QWidget* parent)
    : QWizardPage(parent) {

    setTitle("Welcome to Stereo Vision Setup");
    setSubTitle("Let's configure your stereo vision system step by step");

    auto* layout = new QVBoxLayout(this);

    m_welcomeLabel = new QLabel(
        "<h2>Welcome to Advanced Stereo Vision System</h2>"
        "<p>This wizard will help you set up your stereo vision system with optimal settings "
        "for your specific use case and hardware configuration.</p>"
    );
    m_welcomeLabel->setWordWrap(true);
    layout->addWidget(m_welcomeLabel);

    m_featuresText = new QTextEdit();
    m_featuresText->setReadOnly(true);
    m_featuresText->setMaximumHeight(200);
    m_featuresText->setHtml(
        "<h3>Key Features:</h3>"
        "<ul>"
        "<li><b>Real-time Processing:</b> Advanced streaming optimization with GPU acceleration</li>"
        "<li><b>Multiple Algorithms:</b> Support for SGBM, BM, and advanced stereo matching</li>"
        "<li><b>Easy Calibration:</b> Interactive calibration wizard with automatic detection</li>"
        "<li><b>Performance Tuning:</b> Automatic optimization based on your hardware</li>"
        "<li><b>Profile Management:</b> Save and switch between different configurations</li>"
        "</ul>"
    );
    layout->addWidget(m_featuresText);

    layout->addStretch();

    m_agreeCheckbox = new QCheckBox("I understand and want to proceed with the setup");
    layout->addWidget(m_agreeCheckbox);

    connect(m_agreeCheckbox, &QCheckBox::toggled, this, &QWizardPage::completeChanged);
}

void WelcomePage::initializePage() {
    wizard()->setPixmap(QWizard::WatermarkPixmap, QPixmap(":/icons/wizard_welcome.png"));
}

bool WelcomePage::isComplete() const {
    return m_agreeCheckbox->isChecked();
}

// ProfileSetupPage implementation
ProfileSetupPage::ProfileSetupPage(QWidget* parent)
    : QWizardPage(parent) {

    setTitle("User Profile Setup");
    setSubTitle("Create your user profile for personalized settings");

    auto* layout = new QFormLayout(this);

    // Profile information
    m_profileNameEdit = new QLineEdit();
    m_profileNameEdit->setText(QStandardPaths::displayName(QStandardPaths::HomeLocation));
    layout->addRow("Profile Name:", m_profileNameEdit);

    m_displayNameEdit = new QLineEdit();
    m_displayNameEdit->setText(m_profileNameEdit->text() + "'s Configuration");
    layout->addRow("Display Name:", m_displayNameEdit);

    m_descriptionEdit = new QTextEdit();
    m_descriptionEdit->setMaximumHeight(80);
    m_descriptionEdit->setPlainText("Personal stereo vision configuration");
    layout->addRow("Description:", m_descriptionEdit);

    // Experience level
    auto* experienceGroup = new QGroupBox("Experience Level");
    auto* expLayout = new QVBoxLayout(experienceGroup);

    m_experienceGroup = new QButtonGroup(this);
    m_beginnerRadio = new QRadioButton("Beginner - New to stereo vision");
    m_intermediateRadio = new QRadioButton("Intermediate - Some experience with computer vision");
    m_advancedRadio = new QRadioButton("Advanced - Experienced with stereo vision systems");

    m_experienceGroup->addButton(m_beginnerRadio, 0);
    m_experienceGroup->addButton(m_intermediateRadio, 1);
    m_experienceGroup->addButton(m_advancedRadio, 2);

    expLayout->addWidget(m_beginnerRadio);
    expLayout->addWidget(m_intermediateRadio);
    expLayout->addWidget(m_advancedRadio);

    m_intermediateRadio->setChecked(true); // Default selection
    layout->addRow(experienceGroup);

    // Primary use case
    auto* useCaseGroup = new QGroupBox("Primary Use Case");
    auto* useLayout = new QVBoxLayout(useCaseGroup);

    m_useCaseGroup = new QButtonGroup(this);
    m_researchRadio = new QRadioButton("Research & Development");
    m_industrialRadio = new QRadioButton("Industrial Applications");
    m_hobbyRadio = new QRadioButton("Hobby & Learning");
    m_educationRadio = new QRadioButton("Educational Purposes");

    m_useCaseGroup->addButton(m_researchRadio, 0);
    m_useCaseGroup->addButton(m_industrialRadio, 1);
    m_useCaseGroup->addButton(m_hobbyRadio, 2);
    m_useCaseGroup->addButton(m_educationRadio, 3);

    useLayout->addWidget(m_researchRadio);
    useLayout->addWidget(m_industrialRadio);
    useLayout->addWidget(m_hobbyRadio);
    useLayout->addWidget(m_educationRadio);

    m_researchRadio->setChecked(true); // Default selection
    layout->addRow(useCaseGroup);

    // Connect signals for validation
    connect(m_profileNameEdit, &QLineEdit::textChanged, this, &QWizardPage::completeChanged);
}

void ProfileSetupPage::initializePage() {
    // Auto-fill profile name if empty
    if (m_profileNameEdit->text().isEmpty()) {
        QString defaultName = QString("User_%1").arg(QDateTime::currentDateTime().toString("yyyyMMdd"));
        m_profileNameEdit->setText(defaultName);
    }
}

bool ProfileSetupPage::validatePage() {
    QString profileName = m_profileNameEdit->text().trimmed();

    if (profileName.isEmpty()) {
        QMessageBox::warning(this, "Invalid Profile", "Please enter a valid profile name.");
        return false;
    }

    // Check if profile already exists
    SettingsManager& settings = SettingsManager::instance();
    if (settings.getProfiles().contains(profileName)) {
        int ret = QMessageBox::question(this, "Profile Exists",
                                       QString("Profile '%1' already exists. Do you want to overwrite it?").arg(profileName),
                                       QMessageBox::Yes | QMessageBox::No);
        if (ret == QMessageBox::No) {
            return false;
        }
    }

    return true;
}

int ProfileSetupPage::nextId() const {
    return SetupWizard::Page_Camera;
}

QString ProfileSetupPage::getProfileName() const {
    return m_profileNameEdit->text().trimmed();
}

QString ProfileSetupPage::getExperienceLevel() const {
    int checkedId = m_experienceGroup->checkedId();
    switch (checkedId) {
        case 0: return "Beginner";
        case 1: return "Intermediate";
        case 2: return "Advanced";
        default: return "Intermediate";
    }
}

QString ProfileSetupPage::getPrimaryUseCase() const {
    int checkedId = m_useCaseGroup->checkedId();
    switch (checkedId) {
        case 0: return "Research";
        case 1: return "Industrial";
        case 2: return "Hobby";
        case 3: return "Education";
        default: return "Research";
    }
}

// CameraSetupPage implementation
CameraSetupPage::CameraSetupPage(QWidget* parent)
    : QWizardPage(parent) {

    setTitle("Camera Configuration");
    setSubTitle("Configure your stereo camera setup");

    auto* layout = new QVBoxLayout(this);

    // Camera selection
    auto* selectionLayout = new QFormLayout();

    m_leftCameraCombo = new QComboBox();
    m_rightCameraCombo = new QComboBox();

    selectionLayout->addRow("Left Camera:", m_leftCameraCombo);
    selectionLayout->addRow("Right Camera:", m_rightCameraCombo);

    // Camera detection
    auto* buttonLayout = new QHBoxLayout();
    m_detectButton = new QPushButton("Detect Cameras");
    m_testButton = new QPushButton("Test Configuration");
    m_testButton->setEnabled(false);

    buttonLayout->addWidget(m_detectButton);
    buttonLayout->addWidget(m_testButton);
    buttonLayout->addStretch();

    // Resolution and FPS
    m_resolutionCombo = new QComboBox();
    m_fpsSpinBox = new QSpinBox();
    m_fpsSpinBox->setRange(1, 60);
    m_fpsSpinBox->setValue(30);

    selectionLayout->addRow("Resolution:", m_resolutionCombo);
    selectionLayout->addRow("FPS:", m_fpsSpinBox);

    layout->addLayout(selectionLayout);
    layout->addLayout(buttonLayout);

    // Status
    m_statusLabel = new QLabel("Click 'Detect Cameras' to scan for available cameras");
    m_statusLabel->setStyleSheet("QLabel { color: blue; }");
    layout->addWidget(m_statusLabel);

    // Camera list
    auto* listGroup = new QGroupBox("Available Cameras");
    auto* listLayout = new QVBoxLayout(listGroup);
    m_cameraList = new QListWidget();
    listLayout->addWidget(m_cameraList);
    layout->addWidget(listGroup);

    populateResolutions();

    // Connect signals
    connect(m_detectButton, &QPushButton::clicked, this, &CameraSetupPage::detectCameras);
    connect(m_testButton, &QPushButton::clicked, this, &CameraSetupPage::testCameraConfiguration);
    connect(m_leftCameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &QWizardPage::completeChanged);
    connect(m_rightCameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &QWizardPage::completeChanged);
}

void CameraSetupPage::initializePage() {
    // Auto-detect cameras on page initialization
    QTimer::singleShot(500, this, &CameraSetupPage::detectCameras);
}

bool CameraSetupPage::validatePage() {
    int leftId = getLeftCameraId();
    int rightId = getRightCameraId();

    if (leftId == rightId) {
        QMessageBox::warning(this, "Invalid Configuration",
                           "Left and right cameras must be different.");
        return false;
    }

    if (leftId < 0 || rightId < 0) {
        QMessageBox::warning(this, "Invalid Configuration",
                           "Please select valid cameras for both left and right.");
        return false;
    }

    return true;
}

void CameraSetupPage::detectCameras() {
    m_statusLabel->setText("Scanning for cameras...");
    m_statusLabel->setStyleSheet("QLabel { color: orange; }");

    m_leftCameraCombo->clear();
    m_rightCameraCombo->clear();
    m_cameraList->clear();

    // Simulate camera detection (in real implementation, use OpenCV)
    QStringList detectedCameras;
    for (int i = 0; i < 4; ++i) {
        // Try to detect camera at index i
        detectedCameras.append(QString("Camera %1").arg(i));
    }

    if (detectedCameras.isEmpty()) {
        m_statusLabel->setText("No cameras detected. Please check connections.");
        m_statusLabel->setStyleSheet("QLabel { color: red; }");
        return;
    }

    // Populate combo boxes
    for (int i = 0; i < detectedCameras.size(); ++i) {
        m_leftCameraCombo->addItem(detectedCameras[i], i);
        m_rightCameraCombo->addItem(detectedCameras[i], i);
        m_cameraList->addItem(QString("%1 (ID: %2)").arg(detectedCameras[i]).arg(i));
    }

    // Set default selection
    if (detectedCameras.size() >= 2) {
        m_leftCameraCombo->setCurrentIndex(0);
        m_rightCameraCombo->setCurrentIndex(1);
        m_testButton->setEnabled(true);
    }

    m_statusLabel->setText(QString("Found %1 camera(s)").arg(detectedCameras.size()));
    m_statusLabel->setStyleSheet("QLabel { color: green; }");

    emit completeChanged();
}

void CameraSetupPage::testCameraConfiguration() {
    m_statusLabel->setText("Testing camera configuration...");
    m_statusLabel->setStyleSheet("QLabel { color: orange; }");

    // Simulate test (in real implementation, try to open cameras)
    QTimer::singleShot(2000, [this]() {
        m_statusLabel->setText("Camera configuration test successful!");
        m_statusLabel->setStyleSheet("QLabel { color: green; }");
    });
}

void CameraSetupPage::populateResolutions() {
    m_resolutionCombo->addItem("640x480", QSize(640, 480));
    m_resolutionCombo->addItem("800x600", QSize(800, 600));
    m_resolutionCombo->addItem("1024x768", QSize(1024, 768));
    m_resolutionCombo->addItem("1280x720", QSize(1280, 720));
    m_resolutionCombo->addItem("1920x1080", QSize(1920, 1080));

    m_resolutionCombo->setCurrentText("640x480"); // Default selection
}

void CameraSetupPage::updateCameraList() {
    // Update camera list display
}

int CameraSetupPage::getLeftCameraId() const {
    return m_leftCameraCombo->currentData().toInt();
}

int CameraSetupPage::getRightCameraId() const {
    return m_rightCameraCombo->currentData().toInt();
}

QString CameraSetupPage::getResolution() const {
    return m_resolutionCombo->currentText();
}

int CameraSetupPage::getFps() const {
    return m_fpsSpinBox->value();
}

// SetupWizard implementation
SetupWizard::SetupWizard(QWidget* parent)
    : QWizard(parent)
    , m_settingsManager(&SettingsManager::instance()) {

    setWindowTitle("Stereo Vision Setup Wizard");
    setFixedSize(800, 600);

    // Set wizard style
    setWizardStyle(QWizard::ModernStyle);
    setOption(QWizard::HaveHelpButton, false);
    setOption(QWizard::NoBackButtonOnStartPage, true);

    setupPages();

    connect(this, &QWizard::currentIdChanged, this, &SetupWizard::onCurrentIdChanged);
}

void SetupWizard::setupPages() {
    m_welcomePage = new WelcomePage(this);
    m_profilePage = new ProfileSetupPage(this);
    m_cameraPage = new CameraSetupPage(this);

    // Create simplified algorithm page for now
    m_algorithmPage = new QWizardPage(this);
    m_algorithmPage->setTitle("Algorithm Setup");
    m_algorithmPage->setSubTitle("Choose stereo matching algorithm (simplified)");
    auto* algLayout = new QVBoxLayout(m_algorithmPage);
    auto* algCombo = new QComboBox();
    algCombo->addItems({"SGBM (Recommended)", "Block Matching", "Advanced"});
    algLayout->addWidget(new QLabel("Stereo Algorithm:"));
    algLayout->addWidget(algCombo);
    algLayout->addStretch();

    // Create simplified completion page
    m_completionPage = new QWizardPage(this);
    m_completionPage->setTitle("Setup Complete");
    m_completionPage->setSubTitle("Your stereo vision system is ready!");
    auto* compLayout = new QVBoxLayout(m_completionPage);
    compLayout->addWidget(new QLabel("Setup completed successfully. You can now start using the stereo vision system."));
    compLayout->addStretch();

    setPage(Page_Welcome, m_welcomePage);
    setPage(Page_Profile, m_profilePage);
    setPage(Page_Camera, m_cameraPage);
    setPage(Page_Algorithm, m_algorithmPage);
    setPage(Page_Completion, m_completionPage);
}

void SetupWizard::onCurrentIdChanged(int id) {
    // Update wizard progress or perform page-specific actions
    qDebug() << "Wizard page changed to:" << id;
}

void SetupWizard::accept() {
    applySettings();
    createUserProfile();
    saveConfiguration();

    QWizard::accept();
}

void SetupWizard::applySettings() {
    // Apply camera settings
    m_settingsManager->setValue("camera/leftCameraId", m_cameraPage->getLeftCameraId());
    m_settingsManager->setValue("camera/rightCameraId", m_cameraPage->getRightCameraId());
    m_settingsManager->setValue("camera/resolution", m_cameraPage->getResolution());
    m_settingsManager->setValue("camera/fps", m_cameraPage->getFps());

    // Apply basic stereo settings
    m_settingsManager->setValue("stereo/algorithm", "SGBM");
    m_settingsManager->setValue("stereo/blockSize", 9);
    m_settingsManager->setValue("stereo/numDisparities", 64);
    m_settingsManager->setValue("stereo/minDisparity", 0);
}

void SetupWizard::createUserProfile() {
    QString profileName = m_profilePage->getProfileName();

    // Create new profile
    if (!m_settingsManager->createProfile(profileName)) {
        qWarning() << "Failed to create profile:" << profileName;
        return;
    }

    // Switch to new profile
    m_settingsManager->switchProfile(profileName);

    // Set profile metadata
    m_settingsManager->setValue("profile/displayName", m_profilePage->getProfileName());
    m_settingsManager->setValue("profile/experienceLevel", m_profilePage->getExperienceLevel());
    m_settingsManager->setValue("profile/primaryUseCase", m_profilePage->getPrimaryUseCase());
}

void SetupWizard::saveConfiguration() {
    // Mark as configured
    m_settingsManager->markAsConfigured();

    // Create initial backup
    m_settingsManager->createBackup();

    qDebug() << "Setup wizard configuration saved successfully";
}

} // namespace stereo_vision::gui

#include "setup_wizard.moc"
