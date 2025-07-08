#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QRect>
#include <QRubberBand>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidget>
#include <opencv2/opencv.hpp>

namespace stereo_vision::gui {

class ImageDisplayWidget : public QWidget {
  Q_OBJECT

public:
  explicit ImageDisplayWidget(QWidget *parent = nullptr);
  ~ImageDisplayWidget();

  void setImage(const cv::Mat &image);
  void setImage(const QString &imagePath);
  void clearImage();

  cv::Mat getImage() const { return m_originalImage; }
  QString getImagePath() const { return m_imagePath; }

  void zoomIn();
  void zoomOut();
  void zoomToFit();
  void zoomToActualSize();
  void resetZoom();

  double getZoomFactor() const { return m_zoomFactor; }
  void setZoomFactor(double factor);

  // Selection functionality
  void enableSelection(bool enable) { m_selectionEnabled = enable; }
  bool isSelectionEnabled() const { return m_selectionEnabled; }
  QRect getSelectionRect() const { return m_selectionRect; }
  void clearSelection();

signals:
  void imageChanged();
  void zoomChanged(double factor);
  void selectionChanged(const QRect &rect);
  void mouseClicked(const QPoint &position);
  void mouseMoved(const QPoint &position);

protected:
  void paintEvent(QPaintEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private slots:
  void updateDisplay();

private:
  QPixmap matToQPixmap(const cv::Mat &mat);
  void updateScrollBars();
  QPoint mapToImage(const QPoint &widgetPos) const;
  QPoint mapFromImage(const QPoint &imagePos) const;

  // Image data
  cv::Mat m_originalImage;
  QPixmap m_pixmap;
  QString m_imagePath;

  // Display properties
  double m_zoomFactor;
  QPoint m_imageOffset;
  bool m_dragging;
  QPoint m_lastPanPoint;

  // Selection
  bool m_selectionEnabled;
  bool m_selecting;
  QPoint m_selectionStart;
  QRect m_selectionRect;
  QRubberBand *m_rubberBand;

  // Constants
  static constexpr double MIN_ZOOM = 0.1;
  static constexpr double MAX_ZOOM = 10.0;
  static constexpr double ZOOM_STEP = 1.2;
};

} // namespace stereo_vision::gui
