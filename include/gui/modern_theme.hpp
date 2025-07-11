#pragma once

#include <QApplication>
#include <QGraphicsDropShadowEffect>
#include <QGraphicsEffect>
#include <QPainter>
#include <QPropertyAnimation>
#include <QStyle>
#include <QStyleOption>
#include <QWidget>

namespace stereo_vision::gui {

/**
 * Modern Windows 11-style theme and performance optimization utilities
 */
class ModernTheme {
public:
  // Windows 11 Color Palette
  static const QColor ACCENT_BLUE;
  static const QColor SURFACE_PRIMARY;
  static const QColor SURFACE_SECONDARY;
  static const QColor SURFACE_TERTIARY;
  static const QColor TEXT_PRIMARY;
  static const QColor TEXT_SECONDARY;
  static const QColor BORDER_COLOR;
  static const QColor SELECTION_COLOR;
  static const QColor HOVER_COLOR;

  // Apply Windows 11 style theme to application
  static void applyTheme(QApplication *app);

  // Create modern styled button
  static QString getButtonStyle(bool isPrimary = false);

  // Create modern styled input field
  static QString getInputStyle();

  // Create modern styled panel
  static QString getPanelStyle();

  // Create modern styled tab widget
  static QString getTabStyle();

  // Add drop shadow effect
  static QGraphicsDropShadowEffect *createDropShadow();

  // Create fade animation
  static QPropertyAnimation *createFadeAnimation(QWidget *target);

  // Apply rounded corners
  static void applyRoundedCorners(QWidget *widget, int radius = 8);
};

/**
 * Performance optimization utilities
 */
class PerformanceOptimizer {
public:
  // Enable GPU rendering optimizations
  static void enableGPUAcceleration();

  // Optimize Qt for real-time performance
  static void optimizeQtPerformance();

  // Set high DPI awareness
  static void enableHighDPI();

  // Optimize OpenGL settings
  static void optimizeOpenGL();

  // Memory pool for frequent allocations
  static void setupMemoryPools();
};

/**
 * Modern widget base class with Windows 11 styling
 */
class ModernWidget : public QWidget {
  Q_OBJECT

public:
  explicit ModernWidget(QWidget *parent = nullptr);

protected:
  void paintEvent(QPaintEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;

private:
  bool m_hovered = false;
  QPropertyAnimation *m_hoverAnimation;
  qreal m_hoverOpacity = 0.0;
};

/**
 * High-performance image widget with GPU acceleration
 */
class AcceleratedImageWidget : public QWidget {
  Q_OBJECT

public:
  explicit AcceleratedImageWidget(QWidget *parent = nullptr);
  ~AcceleratedImageWidget();

  void setImage(const cv::Mat &image);
  void setImageAsync(const cv::Mat &image); // Non-blocking version

  // Performance settings
  void setUseGPUAcceleration(bool enabled);
  void setUpdateThrottle(int ms = 16); // ~60 FPS max

protected:
  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private slots:
  void updateDisplay();

private:
  struct Private;
  std::unique_ptr<Private> d;
};

} // namespace stereo_vision::gui
