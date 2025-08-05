#include "gui/performance_benchmark.hpp"
#include <QVBoxLayout>
#include <QLabel>
#include <QTimer>
#include <QPropertyAnimation>

namespace stereo_vision::gui {

struct AutoPerformanceOptimizer::OptimizerPrivate {
  bool optimizationEnabled = false;
  int currentIteration = 0;
  double bestScore = 0.0;
  QTimer* updateTimer = nullptr;
};

AutoPerformanceOptimizer::~AutoPerformanceOptimizer() = default;

} // namespace stereo_vision::gui
