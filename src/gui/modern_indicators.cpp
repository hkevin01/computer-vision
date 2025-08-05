#include "gui/modern_calibration_wizard.hpp"
#include <QVBoxLayout>
#include <QLabel>
#include <QPropertyAnimation>
#include <QColor>
#include <QTimer>

namespace stereo_vision::gui {

struct ModernStepIndicator::StepPrivate {
  QString stepName;
  int stepNumber = 0;
  bool completed = false;
};

struct ModernProgressRing::RingPrivate {
  double progress = 0.0;
  QColor progressColor = Qt::blue;
  QTimer* animationTimer = nullptr;
  QPropertyAnimation* rotationAnimation = nullptr;
};

// Progress indicator is now part of the Progress Ring implementation

} // namespace stereo_vision::gui
