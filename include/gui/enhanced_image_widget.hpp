#pragma once

#include "gui/image_display_widget.hpp"
#include <QMutex>
#include <QThread>
#include <QTimer>
#include <QWidget>
#include <memory>
#include <opencv2/core.hpp>

namespace stereo_vision::gui {

/**
 * Enhanced image display widget with Windows 11 styling and performance
 * optimizations
 */
class EnhancedImageWidget : public QWidget {
  Q_OBJECT

public:
  explicit EnhancedImageWidget(QWidget *parent = nullptr);
  ~EnhancedImageWidget();

  // High-performance image setting
  void setImage(const cv::Mat &image);
  void setImageAsync(const cv::Mat &image);
  void clearImage();

  // Performance settings
  void setMaxFPS(int fps = 60);
  void setUseGPUAcceleration(bool enabled = true);
  void setQualityMode(int mode = 1); // 0=Fast, 1=Balanced, 2=High

  // Modern UI features
  void setCornerRadius(int radius = 8);
  void setDropShadowEnabled(bool enabled = true);
  void setHoverEffectEnabled(bool enabled = true);

  // Zoom and pan with smooth animations
  void zoomIn(bool animated = true);
  void zoomOut(bool animated = true);
  void zoomToFit(bool animated = true);
  void resetZoom(bool animated = true);

  // Selection with modern styling
  void enableSelection(bool enabled);
  QRect getSelection() const;

  // Performance metrics
  struct PerformanceMetrics {
    double avgFPS = 0.0;
    int droppedFrames = 0;
    double renderTime = 0.0;
    int memoryUsage = 0;
  };
  PerformanceMetrics getPerformanceMetrics() const;

signals:
  void imageChanged();
  void zoomChanged(double factor);
  void selectionChanged(const QRect &selection);
  void performanceUpdated(const PerformanceMetrics &metrics);

protected:
  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;

private slots:
  void updateDisplay();
  void onAnimationFinished();
  void updatePerformanceMetrics();

private:
  struct Private;
  std::unique_ptr<Private> d;
};

/**
 * Thread-safe image processing worker for background operations
 */
class ImageProcessor : public QThread {
  Q_OBJECT

public:
  explicit ImageProcessor(QObject *parent = nullptr);
  ~ImageProcessor();

  void processImage(const cv::Mat &image, QSize targetSize, int quality);
  void stop();

signals:
  void imageReady(const QPixmap &pixmap);

protected:
  void run() override;

private:
  struct ProcessorPrivate;
  std::unique_ptr<ProcessorPrivate> d;
};

/**
 * Modern progress indicator for image loading
 */
class ModernProgressIndicator : public QWidget {
  Q_OBJECT

public:
  explicit ModernProgressIndicator(QWidget *parent = nullptr);
  ~ModernProgressIndicator();

  void setProgress(double progress); // 0.0 to 1.0
  void setIndeterminate(bool enabled);
  void setColor(const QColor &color);

protected:
  void paintEvent(QPaintEvent *event) override;

private slots:
  void updateAnimation();

private:
  struct IndicatorPrivate;
  std::unique_ptr<IndicatorPrivate> d;
};

} // namespace stereo_vision::gui
