#include "gui/modern_calibration_wizard.hpp"
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPropertyAnimation>
#include <QGraphicsBlurEffect>
#include <QTimer>

using namespace stereo_vision::gui;

namespace {
struct ModernCalibrationWizardPrivate {
    ModernCalibrationWizard::CalibrationMode mode = ModernCalibrationWizard::ManualMode;
    bool glassEffectEnabled = false;
    bool animationsEnabled = true;
    bool progressAnimationEnabled = true;
    ModernCalibrationWizard::CalibrationMetrics metrics;

    QVBoxLayout* mainLayout = nullptr;
    QLabel* statusLabel = nullptr;
    QPropertyAnimation* progressAnimation = nullptr;
    QGraphicsBlurEffect* glassEffect = nullptr;

    // UI Elements
    ModernStepIndicator* stepIndicator = nullptr;
    ModernProgressRing* progressRing = nullptr;
    ModernCameraPreview* preview = nullptr;

    // State
    int currentStep = 0;
    bool isCalibrating = false;
};
} // anonymous namespace

class ModernCalibrationWizard::Private : public ModernCalibrationWizardPrivate {};

ModernCalibrationWizard::ModernCalibrationWizard(QWidget *parent)
    : QDialog(parent), d(new Private) {
    d->mainLayout = new QVBoxLayout(this);
    d->statusLabel = new QLabel(this);
    d->mainLayout->addWidget(d->statusLabel);

    d->glassEffect = new QGraphicsBlurEffect(this);
    d->glassEffect->setBlurRadius(0);
    d->glassEffect->setEnabled(false);

    d->progressAnimation = new QPropertyAnimation(this, "windowOpacity");
    d->progressAnimation->setDuration(500);
    d->progressAnimation->setStartValue(0.0);
    d->progressAnimation->setEndValue(1.0);

    d->preview = new ModernCameraPreview(this);
    d->stepIndicator = new ModernStepIndicator(this);
    d->progressRing = new ModernProgressRing(this);

    d->mainLayout->addWidget(d->preview);
    d->mainLayout->addWidget(d->stepIndicator);
    d->mainLayout->addWidget(d->progressRing);

    setWindowTitle(tr("Modern Calibration Wizard"));
    resize(800, 600);
}

ModernCalibrationWizard::~ModernCalibrationWizard() = default;

void ModernCalibrationWizard::setCalibrationMode(CalibrationMode mode) {
    if (d->mode != mode) {
        d->mode = mode;
        emit modeChanged(mode);
    }
}

ModernCalibrationWizard::CalibrationMode ModernCalibrationWizard::getCalibrationMode() const {
    return d->mode;
}

void ModernCalibrationWizard::setGlassEffect(bool enabled) {
    d->glassEffectEnabled = enabled;
    d->glassEffect->setEnabled(enabled);
}

void ModernCalibrationWizard::setAnimationsEnabled(bool enabled) {
    d->animationsEnabled = enabled;
    if (d->progressAnimation) {
        d->progressAnimation->setDuration(enabled ? 500 : 0);
    }
}

void ModernCalibrationWizard::setProgressAnimation(bool enabled) {
    d->progressAnimationEnabled = enabled;
}

ModernCalibrationWizard::CalibrationMetrics ModernCalibrationWizard::getCalibrationMetrics() const {
    return d->metrics;
}

void ModernCalibrationWizard::paintEvent(QPaintEvent *event) {
    QDialog::paintEvent(event);
}

void ModernCalibrationWizard::enterEvent(QEvent *event) {
    QDialog::enterEvent(event);
    if (d->animationsEnabled) {
        // Subtle hover effect animation
        QPropertyAnimation *anim = new QPropertyAnimation(this, "windowOpacity");
        anim->setDuration(200);
        anim->setStartValue(0.97);
        anim->setEndValue(1.0);
        anim->start(QPropertyAnimation::DeleteWhenStopped);
    }
}

void ModernCalibrationWizard::leaveEvent(QEvent *event) {
    QDialog::leaveEvent(event);
    if (d->animationsEnabled) {
        // Restore normal opacity
        QPropertyAnimation *anim = new QPropertyAnimation(this, "windowOpacity");
        anim->setDuration(200);
        anim->setStartValue(1.0);
        anim->setEndValue(0.97);
        anim->start(QPropertyAnimation::DeleteWhenStopped);
    }
}

void ModernCalibrationWizard::mousePressEvent(QMouseEvent *event) {
    QDialog::mousePressEvent(event);
}
