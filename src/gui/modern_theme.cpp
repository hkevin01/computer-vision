#include "gui/modern_theme.hpp"
#include <QApplication>
#include <QEasingCurve>
#include <QGraphicsDropShadowEffect>
#include <QMutex>
#include <QOpenGLWidget>
#include <QPainter>
#include <QPixmap>
#include <QPropertyAnimation>
#include <QQueue>
#include <QStyleOption>
#include <QThread>
#include <QTimer>
#include <QWidget>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

// Windows 11 Color Palette
const QColor ModernTheme::ACCENT_BLUE = QColor(0, 120, 212);
const QColor ModernTheme::SURFACE_PRIMARY = QColor(243, 243, 243);
const QColor ModernTheme::SURFACE_SECONDARY = QColor(249, 249, 249);
const QColor ModernTheme::SURFACE_TERTIARY = QColor(255, 255, 255);
const QColor ModernTheme::TEXT_PRIMARY = QColor(32, 31, 30);
const QColor ModernTheme::TEXT_SECONDARY = QColor(96, 94, 92);
const QColor ModernTheme::BORDER_COLOR = QColor(200, 198, 196);
const QColor ModernTheme::SELECTION_COLOR = QColor(0, 120, 212, 40);
const QColor ModernTheme::HOVER_COLOR = QColor(0, 0, 0, 10);

void ModernTheme::applyTheme(QApplication *app) {
  // Windows 11 modern style sheet
  QString styleSheet = R"(
        QMainWindow {
            background-color: #f3f3f3;
            color: #201f1e;
            font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
            font-size: 9pt;
        }
        
        QMenuBar {
            background-color: #f9f9f9;
            border: none;
            border-bottom: 1px solid #c8c6c4;
            padding: 4px 8px;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 2px;
        }
        
        QMenuBar::item:selected {
            background-color: #f0f0f0;
        }
        
        QMenu {
            background-color: #ffffff;
            border: 1px solid #c8c6c4;
            border-radius: 8px;
            padding: 4px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        }
        
        QMenu::item {
            padding: 8px 16px;
            border-radius: 4px;
            margin: 1px;
        }
        
        QMenu::item:selected {
            background-color: #f0f0f0;
        }
        
        QStatusBar {
            background-color: #f9f9f9;
            border-top: 1px solid #c8c6c4;
            padding: 4px;
        }
        
        QToolBar {
            background-color: #f9f9f9;
            border: none;
            spacing: 4px;
            padding: 4px;
        }
        
        QTabWidget::pane {
            border: 1px solid #c8c6c4;
            background-color: #ffffff;
            border-radius: 8px;
            margin-top: -1px;
        }
        
        QTabBar::tab {
            background-color: #f3f3f3;
            color: #605e5c;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            min-width: 80px;
        }
        
        QTabBar::tab:selected {
            background-color: #ffffff;
            color: #201f1e;
            border-bottom: 2px solid #0078d4;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #f0f0f0;
        }
        
        QSplitter::handle {
            background-color: #c8c6c4;
            width: 1px;
            height: 1px;
        }
        
        QSplitter::handle:hover {
            background-color: #0078d4;
        }
    )";

  app->setStyleSheet(styleSheet);

  // Enable modern features
  PerformanceOptimizer::enableHighDPI();
  PerformanceOptimizer::optimizeQtPerformance();
}

QString ModernTheme::getButtonStyle(bool isPrimary) {
  if (isPrimary) {
    return R"(
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #c8c6c4;
                color: #a19f9d;
            }
        )";
  } else {
    return R"(
            QPushButton {
                background-color: #ffffff;
                color: #201f1e;
                border: 1px solid #c8c6c4;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border-color: #a19f9d;
            }
            QPushButton:pressed {
                background-color: #e5e5e5;
            }
            QPushButton:disabled {
                background-color: #f3f3f3;
                color: #a19f9d;
                border-color: #edebe9;
            }
        )";
  }
}

QString ModernTheme::getInputStyle() {
  return R"(
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #ffffff;
            border: 1px solid #c8c6c4;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 9pt;
            selection-background-color: #0078d4;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #0078d4;
            outline: none;
        }
        QComboBox::drop-down {
            border: none;
            border-radius: 0px 4px 4px 0px;
            background-color: transparent;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: url(:/icons/dropdown.png);
            width: 12px;
            height: 12px;
        }
    )";
}

QString ModernTheme::getPanelStyle() {
  return R"(
        QGroupBox {
            background-color: #ffffff;
            border: 1px solid #c8c6c4;
            border-radius: 8px;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
            color: #201f1e;
        }
        QWidget {
            background-color: #f9f9f9;
        }
    )";
}

QString ModernTheme::getTabStyle() {
  return R"(
        QTabWidget::pane {
            border: 1px solid #c8c6c4;
            background-color: #ffffff;
            border-radius: 8px;
            margin-top: -1px;
        }
        QTabBar::tab {
            background-color: #f3f3f3;
            color: #605e5c;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            min-width: 80px;
            font-weight: 500;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            color: #201f1e;
            border-bottom: 2px solid #0078d4;
        }
        QTabBar::tab:hover:!selected {
            background-color: #f0f0f0;
        }
    )";
}

QGraphicsDropShadowEffect *ModernTheme::createDropShadow() {
  auto *shadow = new QGraphicsDropShadowEffect();
  shadow->setColor(QColor(0, 0, 0, 30));
  shadow->setBlurRadius(16);
  shadow->setOffset(0, 4);
  return shadow;
}

QPropertyAnimation *ModernTheme::createFadeAnimation(QWidget *target) {
  auto *animation = new QPropertyAnimation(target, "windowOpacity", target);
  animation->setDuration(200);
  animation->setEasingCurve(QEasingCurve::OutCubic);
  return animation;
}

void ModernTheme::applyRoundedCorners(QWidget *widget, int radius) {
  widget->setStyleSheet(widget->styleSheet() +
                        QString("border-radius: %1px;").arg(radius));
}

// Performance Optimizer Implementation
void PerformanceOptimizer::enableGPUAcceleration() {
  // Enable OpenGL rendering
  QApplication::setAttribute(Qt::AA_UseOpenGLES, true);
  QApplication::setAttribute(Qt::AA_UseSoftwareOpenGL, false);

  // Enable smooth pixmap transforms
  QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps, true);
}

void PerformanceOptimizer::optimizeQtPerformance() {
  // Disable unnecessary animations for performance
  QApplication::setEffectEnabled(Qt::UI_AnimateMenu, false);
  QApplication::setEffectEnabled(Qt::UI_AnimateCombo, false);
  QApplication::setEffectEnabled(Qt::UI_AnimateTooltip, false);

  // Enable sharing OpenGL contexts
  QApplication::setAttribute(Qt::AA_ShareOpenGLContexts, true);

  // Optimize rendering
  QApplication::setAttribute(Qt::AA_CompressHighFrequencyEvents, true);
  QApplication::setAttribute(Qt::AA_CompressTabletEvents, true);
}

void PerformanceOptimizer::enableHighDPI() {
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
  QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps, true);
  QApplication::setHighDpiScaleFactorRoundingPolicy(
      Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
}

void PerformanceOptimizer::optimizeOpenGL() {
  // Set OpenGL format
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  format.setVersion(3, 3);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setSamples(4); // 4x MSAA
  format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
  QSurfaceFormat::setDefaultFormat(format);
}

void PerformanceOptimizer::setupMemoryPools() {
  // This would set up memory pools for frequent allocations
  // Implementation depends on specific memory management needs
}

// Modern Widget Implementation
ModernWidget::ModernWidget(QWidget *parent)
    : QWidget(parent),
      m_hoverAnimation(new QPropertyAnimation(this, "hoverOpacity", this)) {

  m_hoverAnimation->setDuration(150);
  m_hoverAnimation->setEasingCurve(QEasingCurve::OutCubic);

  connect(m_hoverAnimation, &QPropertyAnimation::valueChanged, this,
          [this]() { update(); });
}

void ModernWidget::paintEvent(QPaintEvent *event) {
  QStyleOption opt;
  opt.initFrom(this);
  QPainter painter(this);

  // Draw background with hover effect
  if (m_hoverOpacity > 0.0) {
    painter.fillRect(rect(), QColor(0, 0, 0, int(m_hoverOpacity * 25)));
  }

  style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);
}

void ModernWidget::enterEvent(QEnterEvent *event) {
  m_hovered = true;
  m_hoverAnimation->setStartValue(m_hoverOpacity);
  m_hoverAnimation->setEndValue(1.0);
  m_hoverAnimation->start();
  QWidget::enterEvent(event);
}

void ModernWidget::leaveEvent(QEvent *event) {
  m_hovered = false;
  m_hoverAnimation->setStartValue(m_hoverOpacity);
  m_hoverAnimation->setEndValue(0.0);
  m_hoverAnimation->start();
  QWidget::leaveEvent(event);
}

// Accelerated Image Widget Implementation
struct AcceleratedImageWidget::Private {
  cv::Mat currentImage;
  QPixmap pixmapCache;
  QTimer *updateTimer;
  QMutex imageMutex;
  QQueue<cv::Mat> imageQueue;
  bool useGPUAcceleration = true;
  bool imageChanged = false;

  Private(AcceleratedImageWidget *parent) {
    updateTimer = new QTimer(parent);
    updateTimer->setSingleShot(false);
    updateTimer->setInterval(16); // ~60 FPS
    QObject::connect(updateTimer, &QTimer::timeout, parent,
                     &AcceleratedImageWidget::updateDisplay);
    updateTimer->start();
  }
};

AcceleratedImageWidget::AcceleratedImageWidget(QWidget *parent)
    : QWidget(parent), d(std::make_unique<Private>(this)) {

  setMinimumSize(300, 200);
  setAttribute(Qt::WA_OpaquePaintEvent, true);
  setAttribute(Qt::WA_NoSystemBackground, true);
}

AcceleratedImageWidget::~AcceleratedImageWidget() = default;

void AcceleratedImageWidget::setImage(const cv::Mat &image) {
  QMutexLocker locker(&d->imageMutex);
  d->currentImage = image.clone();
  d->imageChanged = true;
}

void AcceleratedImageWidget::setImageAsync(const cv::Mat &image) {
  QMutexLocker locker(&d->imageMutex);
  d->imageQueue.enqueue(image.clone());
  if (d->imageQueue.size() > 3) {
    d->imageQueue.dequeue(); // Drop old frames
  }
}

void AcceleratedImageWidget::setUseGPUAcceleration(bool enabled) {
  d->useGPUAcceleration = enabled;
}

void AcceleratedImageWidget::setUpdateThrottle(int ms) {
  d->updateTimer->setInterval(ms);
}

void AcceleratedImageWidget::updateDisplay() {
  QMutexLocker locker(&d->imageMutex);

  // Process queued images
  if (!d->imageQueue.isEmpty()) {
    d->currentImage = d->imageQueue.dequeue();
    d->imageChanged = true;
  }

  if (d->imageChanged && !d->currentImage.empty()) {
    // Convert to QPixmap with GPU acceleration if available
    cv::Mat displayImage;
    if (d->currentImage.channels() == 3) {
      cv::cvtColor(d->currentImage, displayImage, cv::COLOR_BGR2RGB);
    } else {
      displayImage = d->currentImage;
    }

    QImage qimg(displayImage.data, displayImage.cols, displayImage.rows,
                displayImage.step, QImage::Format_RGB888);

    // Scale to widget size efficiently
    QSize targetSize = size();
    if (qimg.size() != targetSize) {
      qimg = qimg.scaled(targetSize, Qt::KeepAspectRatio,
                         Qt::SmoothTransformation);
    }

    d->pixmapCache = QPixmap::fromImage(qimg);
    d->imageChanged = false;

    update();
  }
}

void AcceleratedImageWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, false); // Faster for video
  painter.setRenderHint(QPainter::SmoothPixmapTransform, d->useGPUAcceleration);

  painter.fillRect(rect(), Qt::black);

  if (!d->pixmapCache.isNull()) {
    QRect drawRect = rect();
    QRect pixmapRect = d->pixmapCache.rect();

    // Center the image
    int x = (drawRect.width() - pixmapRect.width()) / 2;
    int y = (drawRect.height() - pixmapRect.height()) / 2;

    painter.drawPixmap(x, y, d->pixmapCache);
  } else {
    painter.setPen(Qt::white);
    painter.drawText(rect(), Qt::AlignCenter, "No image");
  }
}

void AcceleratedImageWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  QMutexLocker locker(&d->imageMutex);
  d->imageChanged = true; // Force redraw at new size
}

} // namespace stereo_vision::gui
