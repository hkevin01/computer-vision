#pragma once

#include "gui/calibration_wizard.hpp"
#include "gui/modern_theme.hpp"
#include <QDialog>
#include <QGraphicsEffect>
#include <QPropertyAnimation>
#include <memory>

namespace stereo_vision::gui {

/**
 * Modern Windows 11-styled calibration wizard with enhanced UX
 */
class ModernCalibrationWizard : public QDialog {
  Q_OBJECT

public:
  explicit ModernCalibrationWizard(QWidget *parent = nullptr);
  ~ModernCalibrationWizard();

  // Enhanced calibration workflow
  enum CalibrationMode { ManualMode, AutoMode, AIAssistedMode };

  void setCalibrationMode(CalibrationMode mode);
  CalibrationMode getCalibrationMode() const;

  // Modern UI features
  void setGlassEffect(bool enabled);
  void setAnimationsEnabled(bool enabled);
  void setProgressAnimation(bool enabled);

  // Performance monitoring
  struct CalibrationMetrics {
    int framesProcessed = 0;
    double averageReprojectionError = 0.0;
    double processingTime = 0.0;
    int detectionRate = 0; // Percentage
    QString qualityAssessment;
  };

  CalibrationMetrics getCalibrationMetrics() const;

signals:
  void calibrationProgress(int percentage);
  void calibrationQualityChanged(const QString &quality);
  void calibrationCompleted(bool success);
  void modeChanged(CalibrationMode mode);

protected:
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
  void paintEvent(QPaintEvent *event) override;
  void keyPressEvent(QKeyEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;

private slots:
  void onStepChanged(int step);
  void onCalibrationProgress(int progress);
  void onAnimationFinished();
  void updateQualityAssessment();
  void updateOverlay();
  void animateQualityIndicator();

private:
  void setupModernUI();
  void setupAnimations();
  void setupGlassEffect();
  void applyWindowsStyle();
  void createProgressIndicator();
  void createModernButtons();
  void createStepIndicator();

  struct Private;
  std::unique_ptr<Private> d;
};

/**
 * Modern step indicator with smooth animations
 */
class ModernStepIndicator : public QWidget {
  Q_OBJECT

public:
  explicit ModernStepIndicator(QWidget *parent = nullptr);
  ~ModernStepIndicator();

  void setStepCount(int count);
  void setCurrentStep(int step);
  void setStepTitle(int step, const QString &title);
  void setStepComplete(int step, bool complete);

  int getCurrentStep() const;
  int getStepCount() const;

signals:
  void stepClicked(int step);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private slots:
  void updateAnimation();

private:
  struct StepPrivate;
  std::unique_ptr<StepPrivate> d;
};

/**
 * Modern progress ring with smooth animations
 */
class ModernProgressRing : public QWidget {
  Q_OBJECT
  Q_PROPERTY(double progress READ progress WRITE setProgress)
  Q_PROPERTY(double animationProgress READ animationProgress WRITE
                 setAnimationProgress)

public:
  explicit ModernProgressRing(QWidget *parent = nullptr);
  ~ModernProgressRing();

  double progress() const;
  void setProgress(double progress, bool animated = true);

  void setRingColor(const QColor &color);
  void setBackgroundColor(const QColor &color);
  void setTextColor(const QColor &color);

  void setText(const QString &text);
  void setSubText(const QString &text);

protected:
  void paintEvent(QPaintEvent *event) override;

private:
  double animationProgress() const;
  void setAnimationProgress(double progress);

  struct RingPrivate;
  std::unique_ptr<RingPrivate> d;
};

/**
 * Enhanced camera preview with modern styling
 */
class ModernCameraPreview : public QWidget {
  Q_OBJECT

public:
  explicit ModernCameraPreview(QWidget *parent = nullptr);
  ~ModernCameraPreview();

  void setImage(const cv::Mat &image);
  void setOverlayEnabled(bool enabled);
  void setPatternDetection(bool enabled);
  void setQualityIndicator(bool enabled);

  // Quality assessment
  enum QualityLevel { Poor, Fair, Good, Excellent };

  void setQualityLevel(QualityLevel level);
  QualityLevel getQualityLevel() const;

  // Pattern detection visualization
  void setDetectedPattern(const std::vector<cv::Point2f> &corners, bool valid);
  void
  setPatternType(int type); // 0=chessboard, 1=circles, 2=asymmetric circles

signals:
  void qualityChanged(QualityLevel level);
  void patternDetected(bool valid);
  void captureRequested();

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void enterEvent(QEvent *event) override;
  void leaveEvent(QEvent *event) override;

private slots:
  void updateOverlay();
  void animateQualityIndicator();

private:
  struct Private;
  std::unique_ptr<Private> d;

private:
  struct PreviewPrivate;
  const std::unique_ptr<PreviewPrivate> preview_d;
};

} // namespace stereo_vision::gui
