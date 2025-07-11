#pragma once

#include <QElapsedTimer>
#include <QMap>
#include <QMouseEvent>
#include <QObject>
#include <QPaintEvent>
#include <QScopeGuard>
#include <QString>
#include <QTimer>
#include <QWidget>
#include <memory>

namespace stereo_vision::gui {

/**
 * Performance monitoring and optimization utility for Windows 11 standards
 */
class PerformanceBenchmark : public QObject {
  Q_OBJECT

public:
  static PerformanceBenchmark *instance();

  // Performance metrics
  struct Metrics {
    double fps = 0.0;
    double cpuUsage = 0.0;
    double memoryUsage = 0.0; // MB
    double gpuUsage = 0.0;
    double renderTime = 0.0; // ms
    double ioTime = 0.0;     // ms
    int droppedFrames = 0;
    QString status = "Optimal";
  };

  // Benchmark categories
  enum BenchmarkType {
    UI_Rendering,
    Image_Processing,
    Camera_Capture,
    Calibration_Processing,
    Point_Cloud_Generation,
    File_IO,
    Memory_Allocation
  };

  // Performance profiling
  void startProfiling(const QString &name);
  void endProfiling(const QString &name);
  void markFrame();

  // System monitoring
  void startMonitoring();
  void stopMonitoring();
  bool isMonitoring() const;

  // Performance optimization
  void enableOptimizations();
  void optimizeForRealTime();
  void optimizeForQuality();
  void optimizeForMemory();

  // Metrics access
  Metrics getCurrentMetrics() const;
  QMap<QString, double> getDetailedMetrics() const;
  QString getPerformanceReport() const;

  // Windows 11 specific optimizations
  void enableWin11Optimizations();
  void setHighPerformanceMode(bool enabled);
  void enableHardwareAcceleration(bool enabled);

signals:
  void metricsUpdated(const Metrics &metrics);
  void performanceWarning(const QString &warning);
  void optimizationSuggestion(const QString &suggestion);

private slots:
  void updateMetrics();
  void checkPerformanceThresholds();

private:
  explicit PerformanceBenchmark(QObject *parent = nullptr);
  ~PerformanceBenchmark();

  struct Private;
  std::unique_ptr<Private> d;
};

/**
 * Real-time performance widget for monitoring during development
 */
class PerformanceWidget : public QWidget {
  Q_OBJECT

public:
  explicit PerformanceWidget(QWidget *parent = nullptr);
  ~PerformanceWidget();

  void setVisible(bool visible) override;
  void setUpdateInterval(int ms);
  void setCompactMode(bool compact);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;

private slots:
  void onMetricsUpdated(const PerformanceBenchmark::Metrics &metrics);
  void updateDisplay();

private:
  struct WidgetPrivate;
  std::unique_ptr<WidgetPrivate> d;
};

/**
 * Automatic performance optimizer that adjusts settings based on system
 * capabilities
 */
class AutoPerformanceOptimizer : public QObject {
  Q_OBJECT

public:
  explicit AutoPerformanceOptimizer(QObject *parent = nullptr);

  // System capability detection
  struct SystemCapabilities {
    bool hasGPU = false;
    bool hasMultipleGPUs = false;
    int cpuCores = 1;
    double memoryGB = 1.0;
    bool isHighEndSystem = false;
    QString gpuVendor;
    QString recommendations;
  };

  SystemCapabilities detectCapabilities();
  void applyOptimalSettings();
  void enableAdaptiveOptimization(bool enabled);

  // Quality vs Performance trade-offs
  enum OptimizationProfile {
    MaximumQuality,
    Balanced,
    MaximumPerformance,
    PowerSaving,
    Adaptive
  };

  void setOptimizationProfile(OptimizationProfile profile);
  OptimizationProfile getOptimizationProfile() const;

signals:
  void optimizationApplied(const QString &description);
  void profileChanged(OptimizationProfile profile);

private slots:
  void onPerformanceChanged(const PerformanceBenchmark::Metrics &metrics);
  void adaptSettings();

private:
  struct OptimizerPrivate;
  std::unique_ptr<OptimizerPrivate> d;
};

// Convenience macros for performance profiling
#define PERF_PROFILE(name)                                                     \
  stereo_vision::gui::PerformanceBenchmark::instance()->startProfiling(name);  \
  QScopeGuard guard([]() {                                                     \
    stereo_vision::gui::PerformanceBenchmark::instance()->endProfiling(name);  \
  });

#define PERF_MARK_FRAME()                                                      \
  stereo_vision::gui::PerformanceBenchmark::instance()->markFrame();

} // namespace stereo_vision::gui
