#include "gui/enhanced_image_widget.hpp"
#include "gui/modern_theme.hpp"
#include <QApplication>
#include <QEasingCurve>
#include <QElapsedTimer>
#include <QGraphicsDropShadowEffect>
#include <QMutex>
#include <QMutexLocker>
#include <QPainter>
#include <QPropertyAnimation>
#include <QQueue>
#include <QScreen>
#include <QTimer>
#include <cmath>

namespace stereo_vision::gui {

// Enhanced Image Widget Private Implementation
struct EnhancedImageWidget::Private {
  // Image data
  cv::Mat currentImage;
  QPixmap displayPixmap;
  QPixmap scaledPixmap;

  // Performance settings
  QTimer *updateTimer;
  QTimer *metricsTimer;
  ImageProcessor *processor;
  QMutex imageMutex;
  QQueue<cv::Mat> imageQueue;

  // Display settings
  double zoomFactor = 1.0;
  QPointF panOffset = QPointF(0, 0);
  QPoint lastPanPoint;
  bool dragging = false;
  int cornerRadius = 8;
  bool dropShadowEnabled = true;
  bool hoverEffectEnabled = true;
  bool useGPUAcceleration = true;
  int qualityMode = 1;

  // Selection
  bool selectionEnabled = false;
  bool selecting = false;
  QPoint selectionStart;
  QRect selectionRect;

  // Animations
  QPropertyAnimation *zoomAnimation;
  QPropertyAnimation *panAnimation;
  QPropertyAnimation *hoverAnimation;
  double hoverOpacity = 0.0;
  bool hovered = false;

  // Performance metrics
  PerformanceMetrics metrics;
  QElapsedTimer frameTimer;
  QQueue<qint64> frameTimes;
  int frameCount = 0;

  Private(EnhancedImageWidget *parent) {
    // Setup timers
    updateTimer = new QTimer(parent);
    updateTimer->setSingleShot(false);
    updateTimer->setInterval(16); // 60 FPS

    metricsTimer = new QTimer(parent);
    metricsTimer->setInterval(1000); // Update metrics every second

    // Setup image processor
    processor = new ImageProcessor(parent);

    // Setup animations
    zoomAnimation = new QPropertyAnimation(parent, "zoomFactor", parent);
    zoomAnimation->setDuration(300);
    zoomAnimation->setEasingCurve(QEasingCurve::OutCubic);

    panAnimation = new QPropertyAnimation(parent, "panOffset", parent);
    panAnimation->setDuration(300);
    panAnimation->setEasingCurve(QEasingCurve::OutCubic);

    hoverAnimation = new QPropertyAnimation(parent, "hoverOpacity", parent);
    hoverAnimation->setDuration(200);
    hoverAnimation->setEasingCurve(QEasingCurve::OutCubic);

    frameTimer.start();
  }
};

EnhancedImageWidget::EnhancedImageWidget(QWidget *parent)
    : QWidget(parent), d(std::make_unique<Private>(this)) {

  setMinimumSize(300, 200);
  setAttribute(Qt::WA_OpaquePaintEvent, true);
  setAttribute(Qt::WA_NoSystemBackground, true);
  setMouseTracking(true);

  // Apply modern styling
  setStyleSheet(QString("EnhancedImageWidget {"
                        "    background-color: %1;"
                        "    border: 1px solid %2;"
                        "    border-radius: %3px;"
                        "}")
                    .arg(ModernTheme::SURFACE_TERTIARY.name())
                    .arg(ModernTheme::BORDER_COLOR.name())
                    .arg(d->cornerRadius));

  // Add drop shadow if enabled
  if (d->dropShadowEnabled) {
    setGraphicsEffect(ModernTheme::createDropShadow());
  }

  // Connect signals
  connect(d->updateTimer, &QTimer::timeout, this,
          &EnhancedImageWidget::updateDisplay);
  connect(d->metricsTimer, &QTimer::timeout, this,
          &EnhancedImageWidget::updatePerformanceMetrics);
  connect(d->processor, &ImageProcessor::imageReady, this,
          [this](const QPixmap &pixmap) {
            d->displayPixmap = pixmap;
            update();
          });

  // Connect animations
  connect(d->zoomAnimation, &QPropertyAnimation::valueChanged, this,
          [this]() { update(); });
  connect(d->panAnimation, &QPropertyAnimation::valueChanged, this,
          [this]() { update(); });
  connect(d->hoverAnimation, &QPropertyAnimation::valueChanged, this,
          [this]() { update(); });

  // Start timers
  d->updateTimer->start();
  d->metricsTimer->start();
}

EnhancedImageWidget::~EnhancedImageWidget() {
  d->processor->stop();
  d->processor->wait();
}

void EnhancedImageWidget::setImage(const cv::Mat &image) {
  QMutexLocker locker(&d->imageMutex);
  d->currentImage = image.clone();

  if (!image.empty()) {
    // Process image in background thread
    d->processor->processImage(image, size(), d->qualityMode);
  } else {
    d->displayPixmap = QPixmap();
    update();
  }
}

void EnhancedImageWidget::setImageAsync(const cv::Mat &image) {
  QMutexLocker locker(&d->imageMutex);
  d->imageQueue.enqueue(image.clone());

  // Limit queue size to prevent memory issues
  while (d->imageQueue.size() > 3) {
    d->imageQueue.dequeue();
    d->metrics.droppedFrames++;
  }
}

void EnhancedImageWidget::clearImage() {
  QMutexLocker locker(&d->imageMutex);
  d->currentImage = cv::Mat();
  d->displayPixmap = QPixmap();
  d->imageQueue.clear();
  update();
}

void EnhancedImageWidget::setMaxFPS(int fps) {
  d->updateTimer->setInterval(1000 / qMax(1, fps));
}

void EnhancedImageWidget::setUseGPUAcceleration(bool enabled) {
  d->useGPUAcceleration = enabled;
}

void EnhancedImageWidget::setQualityMode(int mode) {
  d->qualityMode = qBound(0, mode, 2);
}

void EnhancedImageWidget::setCornerRadius(int radius) {
  d->cornerRadius = radius;
  setStyleSheet(
      styleSheet().replace(QRegularExpression("border-radius: \\d+px"),
                           QString("border-radius: %1px").arg(radius)));
}

void EnhancedImageWidget::setDropShadowEnabled(bool enabled) {
  d->dropShadowEnabled = enabled;
  if (enabled && !graphicsEffect()) {
    setGraphicsEffect(ModernTheme::createDropShadow());
  } else if (!enabled && graphicsEffect()) {
    setGraphicsEffect(nullptr);
  }
}

void EnhancedImageWidget::setHoverEffectEnabled(bool enabled) {
  d->hoverEffectEnabled = enabled;
}

void EnhancedImageWidget::zoomIn(bool animated) {
  double newZoom = d->zoomFactor * 1.25;
  if (animated) {
    d->zoomAnimation->setStartValue(d->zoomFactor);
    d->zoomAnimation->setEndValue(newZoom);
    d->zoomAnimation->start();
  } else {
    d->zoomFactor = newZoom;
    update();
  }
  emit zoomChanged(d->zoomFactor);
}

void EnhancedImageWidget::zoomOut(bool animated) {
  double newZoom = d->zoomFactor / 1.25;
  if (animated) {
    d->zoomAnimation->setStartValue(d->zoomFactor);
    d->zoomAnimation->setEndValue(newZoom);
    d->zoomAnimation->start();
  } else {
    d->zoomFactor = newZoom;
    update();
  }
  emit zoomChanged(d->zoomFactor);
}

void EnhancedImageWidget::zoomToFit(bool animated) {
  if (d->displayPixmap.isNull())
    return;

  QSize widgetSize = size() - QSize(20, 20); // Account for padding
  QSize pixmapSize = d->displayPixmap.size();

  double scaleX = (double)widgetSize.width() / pixmapSize.width();
  double scaleY = (double)widgetSize.height() / pixmapSize.height();
  double newZoom = qMin(scaleX, scaleY);

  if (animated) {
    d->zoomAnimation->setStartValue(d->zoomFactor);
    d->zoomAnimation->setEndValue(newZoom);
    d->zoomAnimation->start();

    d->panAnimation->setStartValue(d->panOffset);
    d->panAnimation->setEndValue(QPointF(0, 0));
    d->panAnimation->start();
  } else {
    d->zoomFactor = newZoom;
    d->panOffset = QPointF(0, 0);
    update();
  }
  emit zoomChanged(d->zoomFactor);
}

void EnhancedImageWidget::resetZoom(bool animated) {
  if (animated) {
    d->zoomAnimation->setStartValue(d->zoomFactor);
    d->zoomAnimation->setEndValue(1.0);
    d->zoomAnimation->start();

    d->panAnimation->setStartValue(d->panOffset);
    d->panAnimation->setEndValue(QPointF(0, 0));
    d->panAnimation->start();
  } else {
    d->zoomFactor = 1.0;
    d->panOffset = QPointF(0, 0);
    update();
  }
  emit zoomChanged(d->zoomFactor);
}

void EnhancedImageWidget::enableSelection(bool enabled) {
  d->selectionEnabled = enabled;
  if (!enabled) {
    d->selectionRect = QRect();
    d->selecting = false;
    update();
  }
}

QRect EnhancedImageWidget::getSelection() const { return d->selectionRect; }

EnhancedImageWidget::PerformanceMetrics
EnhancedImageWidget::getPerformanceMetrics() const {
  return d->metrics;
}

void EnhancedImageWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);

  // Enable high-quality rendering based on quality mode
  if (d->qualityMode >= 1) {
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform,
                          d->useGPUAcceleration);
  }

  // Record frame time
  qint64 frameTime = d->frameTimer.nsecsElapsed();
  d->frameTimer.restart();

  // Fill background
  painter.fillRect(rect(), ModernTheme::SURFACE_TERTIARY);

  if (!d->displayPixmap.isNull()) {
    // Calculate display rect with zoom and pan
    QSize scaledSize = d->displayPixmap.size() * d->zoomFactor;
    QRect imageRect = QRect(QPoint(0, 0), scaledSize);

    // Center and apply pan offset
    QPoint center = rect().center();
    imageRect.moveCenter(center + d->panOffset.toPoint());

    // Draw the image
    painter.drawPixmap(imageRect, d->displayPixmap);

    // Draw selection if active
    if (d->selectionEnabled && !d->selectionRect.isEmpty()) {
      painter.setPen(QPen(ModernTheme::ACCENT_BLUE, 2));
      painter.setBrush(ModernTheme::SELECTION_COLOR);
      painter.drawRect(d->selectionRect);
    }
  } else {
    // Draw placeholder
    painter.setPen(ModernTheme::TEXT_SECONDARY);
    painter.drawText(rect(), Qt::AlignCenter, "No image loaded");
  }

  // Draw hover effect
  if (d->hoverEffectEnabled && d->hoverOpacity > 0.0) {
    painter.fillRect(rect(), QColor(255, 255, 255, int(d->hoverOpacity * 20)));
  }

  // Update performance metrics
  d->frameTimes.enqueue(frameTime);
  while (d->frameTimes.size() > 60) { // Keep last 60 frames
    d->frameTimes.dequeue();
  }
}

void EnhancedImageWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);

  // Reprocess image at new size if needed
  if (!d->currentImage.empty()) {
    d->processor->processImage(d->currentImage, size(), d->qualityMode);
  }
}

void EnhancedImageWidget::wheelEvent(QWheelEvent *event) {
  if (event->modifiers() & Qt::ControlModifier) {
    // Zoom with mouse wheel
    const double scaleFactor = 1.15;
    if (event->angleDelta().y() > 0) {
      zoomIn(false);
    } else {
      zoomOut(false);
    }
    event->accept();
  } else {
    QWidget::wheelEvent(event);
  }
}

void EnhancedImageWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    if (d->selectionEnabled) {
      d->selecting = true;
      d->selectionStart = event->pos();
      d->selectionRect = QRect();
    } else {
      d->dragging = true;
      d->lastPanPoint = event->pos();
      setCursor(Qt::ClosedHandCursor);
    }
  }
  QWidget::mousePressEvent(event);
}

void EnhancedImageWidget::mouseMoveEvent(QMouseEvent *event) {
  if (d->selecting) {
    d->selectionRect = QRect(d->selectionStart, event->pos()).normalized();
    update();
    emit selectionChanged(d->selectionRect);
  } else if (d->dragging) {
    QPoint delta = event->pos() - d->lastPanPoint;
    d->panOffset += QPointF(delta);
    d->lastPanPoint = event->pos();
    update();
  }
  QWidget::mouseMoveEvent(event);
}

void EnhancedImageWidget::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    if (d->selecting) {
      d->selecting = false;
      emit selectionChanged(d->selectionRect);
    } else if (d->dragging) {
      d->dragging = false;
      setCursor(Qt::ArrowCursor);
    }
  }
  QWidget::mouseReleaseEvent(event);
}

void EnhancedImageWidget::enterEvent(QEvent *event) {
  if (d->hoverEffectEnabled) {
    d->hovered = true;
    d->hoverAnimation->setStartValue(d->hoverOpacity);
    d->hoverAnimation->setEndValue(1.0);
    d->hoverAnimation->start();
  }
  QWidget::enterEvent(event);
}

void EnhancedImageWidget::leaveEvent(QEvent *event) {
  if (d->hoverEffectEnabled) {
    d->hovered = false;
    d->hoverAnimation->setStartValue(d->hoverOpacity);
    d->hoverAnimation->setEndValue(0.0);
    d->hoverAnimation->start();
  }
  QWidget::leaveEvent(event);
}

void EnhancedImageWidget::updateDisplay() {
  QMutexLocker locker(&d->imageMutex);

  // Process queued images
  if (!d->imageQueue.isEmpty()) {
    cv::Mat newImage = d->imageQueue.dequeue();
    d->currentImage = newImage;
    d->processor->processImage(newImage, size(), d->qualityMode);
  }

  d->frameCount++;
}

void EnhancedImageWidget::onAnimationFinished() { update(); }

void EnhancedImageWidget::updatePerformanceMetrics() {
  // Calculate average FPS
  if (!d->frameTimes.isEmpty()) {
    qint64 totalTime = 0;
    for (qint64 time : d->frameTimes) {
      totalTime += time;
    }
    double avgFrameTime = totalTime / double(d->frameTimes.size());
    d->metrics.avgFPS = 1000000000.0 / avgFrameTime; // Convert from nanoseconds
    d->metrics.renderTime = avgFrameTime / 1000000.0; // Convert to milliseconds
  }

  d->metrics.memoryUsage = d->imageQueue.size() * sizeof(cv::Mat);
  emit performanceUpdated(d->metrics);
}

// Image Processor Implementation
struct ImageProcessor::ProcessorPrivate {
  QMutex processMutex;
  QQueue<QPair<cv::Mat, QPair<QSize, int>>> processQueue;
  bool running = true;

  QPixmap convertMatToPixmap(const cv::Mat &mat, QSize targetSize,
                             int quality) {
    if (mat.empty())
      return QPixmap();

    cv::Mat processedMat;

    // Convert color space if needed
    if (mat.channels() == 3) {
      cv::cvtColor(mat, processedMat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
      cv::cvtColor(mat, processedMat, cv::COLOR_GRAY2RGB);
    } else {
      processedMat = mat.clone();
    }

    // Create QImage
    QImage qimg(processedMat.data, processedMat.cols, processedMat.rows,
                processedMat.step, QImage::Format_RGB888);

    // Scale if needed
    if (!targetSize.isEmpty() && qimg.size() != targetSize) {
      Qt::TransformationMode mode =
          (quality >= 2) ? Qt::SmoothTransformation : Qt::FastTransformation;
      qimg = qimg.scaled(targetSize, Qt::KeepAspectRatio, mode);
    }

    return QPixmap::fromImage(qimg);
  }
};

ImageProcessor::ImageProcessor(QObject *parent)
    : QThread(parent), d(std::make_unique<ProcessorPrivate>()) {}

ImageProcessor::~ImageProcessor() {
  stop();
  wait();
}

void ImageProcessor::processImage(const cv::Mat &image, QSize targetSize,
                                  int quality) {
  QMutexLocker locker(&d->processMutex);
  d->processQueue.enqueue(
      qMakePair(image.clone(), qMakePair(targetSize, quality)));

  if (!isRunning()) {
    start();
  }
}

void ImageProcessor::stop() {
  QMutexLocker locker(&d->processMutex);
  d->running = false;
  d->processQueue.clear();
}

void ImageProcessor::run() {
  while (d->running) {
    QPair<cv::Mat, QPair<QSize, int>> task;

    {
      QMutexLocker locker(&d->processMutex);
      if (d->processQueue.isEmpty()) {
        msleep(10);
        continue;
      }
      task = d->processQueue.dequeue();
    }

    QPixmap result = d->convertMatToPixmap(task.first, task.second.first,
                                           task.second.second);
    emit imageReady(result);
  }
}

} // namespace stereo_vision::gui

struct stereo_vision::gui::ModernProgressIndicator::IndicatorPrivate {
    double progress = 0.0;
    bool indeterminate = false;
    QColor color = QColor(0, 120, 215);
    QTimer* animationTimer = nullptr;
    double animationProgress = 0.0;
    int animationDirection = 1;
};

stereo_vision::gui::ModernProgressIndicator::ModernProgressIndicator(QWidget *parent)
    : QWidget(parent), d(std::make_unique<IndicatorPrivate>()) {
    setMinimumHeight(4);
    setMaximumHeight(4);

    d->animationTimer = new QTimer(this);
    connect(d->animationTimer, &QTimer::timeout, this, &ModernProgressIndicator::updateAnimation);
}

// Destructor for PIMPL class
stereo_vision::gui::ModernProgressIndicator::~ModernProgressIndicator() = default;

void stereo_vision::gui::ModernProgressIndicator::setProgress(double progress) {
    d->progress = qBound(0.0, progress, 1.0);
    update();
}

void stereo_vision::gui::ModernProgressIndicator::setIndeterminate(bool enabled) {
    d->indeterminate = enabled;
    if (enabled) {
        d->animationTimer->start(16); // ~60 FPS
    } else {
        d->animationTimer->stop();
    }
    update();
}

void stereo_vision::gui::ModernProgressIndicator::setColor(const QColor &color) {
    d->color = color;
    update();
}

void stereo_vision::gui::ModernProgressIndicator::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    QRect progressRect = rect();

    // Draw background
    painter.fillRect(progressRect, QColor(240, 240, 240));

    if (d->indeterminate) {
        // Draw animated indeterminate progress
        int barWidth = width() / 3;
        int position = static_cast<int>(d->animationProgress * (width() + barWidth)) - barWidth;
        QRect animRect(position, 0, barWidth, height());
        painter.fillRect(animRect.intersected(progressRect), d->color);
    } else {
        // Draw determinate progress
        int progressWidth = static_cast<int>(width() * d->progress);
        QRect fillRect(0, 0, progressWidth, height());
        painter.fillRect(fillRect, d->color);
    }
}

void stereo_vision::gui::ModernProgressIndicator::updateAnimation() {
    d->animationProgress += 0.02 * d->animationDirection;
    if (d->animationProgress >= 1.0) {
        d->animationProgress = 0.0;
    }
    update();
}
