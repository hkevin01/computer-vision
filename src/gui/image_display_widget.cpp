#include "gui/image_display_widget.hpp"
#include <QFileInfo>
#include <QPaintEvent>
#include <QPainter>
#include <QPixmap>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace stereo_vision::gui {

ImageDisplayWidget::ImageDisplayWidget(QWidget *parent)
    : QWidget(parent), m_zoomFactor(1.0), m_dragging(false),
      m_selectionEnabled(false), m_selecting(false), m_rubberBand(nullptr) {
  setMinimumSize(300, 200);
  setStyleSheet("border: 1px solid gray;");
}

ImageDisplayWidget::~ImageDisplayWidget() {
  // Qt handles deletion of child widgets
}

void ImageDisplayWidget::setImage(const QString &imagePath) {
  if (QFileInfo::exists(imagePath)) {
    m_originalImage = cv::imread(imagePath.toStdString());
    m_imagePath = imagePath;
    if (!m_originalImage.empty()) {
      // Convert cv::Mat to QPixmap
      cv::Mat rgbImage;
      if (m_originalImage.channels() == 3) {
        cv::cvtColor(m_originalImage, rgbImage, cv::COLOR_BGR2RGB);
      } else {
        rgbImage = m_originalImage.clone();
      }

      QImage qimg(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step,
                  QImage::Format_RGB888);
      m_pixmap = QPixmap::fromImage(qimg);
      update();
    }
  }
}

void ImageDisplayWidget::setImage(const cv::Mat &image) {
  if (image.empty()) {
    clearImage();
    return;
  }

  m_originalImage = image.clone();
  m_imagePath.clear();

  // Convert cv::Mat to QPixmap
  cv::Mat rgbImage;
  if (image.channels() == 3) {
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
  } else {
    rgbImage = image.clone();
  }

  QImage qimg(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step,
              QImage::Format_RGB888);
  m_pixmap = QPixmap::fromImage(qimg);
  update();
}

void ImageDisplayWidget::clearImage() {
  m_originalImage = cv::Mat();
  m_pixmap = QPixmap();
  m_imagePath.clear();
  clearSelection();
  update();
}

void ImageDisplayWidget::zoomIn() { setZoomFactor(m_zoomFactor * ZOOM_STEP); }

void ImageDisplayWidget::zoomOut() { setZoomFactor(m_zoomFactor / ZOOM_STEP); }

void ImageDisplayWidget::zoomToFit() {
  if (m_pixmap.isNull())
    return;

  double scaleX = static_cast<double>(width()) / m_pixmap.width();
  double scaleY = static_cast<double>(height()) / m_pixmap.height();
  setZoomFactor(std::min(scaleX, scaleY));
}

void ImageDisplayWidget::zoomToActualSize() { setZoomFactor(1.0); }

void ImageDisplayWidget::resetZoom() {
  setZoomFactor(1.0);
  m_imageOffset = QPoint(0, 0);
  update();
}

void ImageDisplayWidget::setZoomFactor(double factor) {
  double newFactor = std::clamp(factor, MIN_ZOOM, MAX_ZOOM);
  if (std::abs(newFactor - m_zoomFactor) > 1e-6) {
    m_zoomFactor = newFactor;
    update();
    emit zoomChanged(m_zoomFactor);
  }
}

void ImageDisplayWidget::clearSelection() {
  if (m_rubberBand) {
    m_rubberBand->hide();
  }
  m_selectionRect = QRect();
  m_selecting = false;
  emit selectionChanged(m_selectionRect);
}

void ImageDisplayWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.fillRect(rect(), Qt::black);

  if (m_pixmap.isNull()) {
    painter.setPen(Qt::white);
    painter.drawText(rect(), Qt::AlignCenter, "No image loaded");
    return;
  }

  // Calculate scaled pixmap size
  QSize scaledSize = m_pixmap.size() * m_zoomFactor;

  // Center the image if it's smaller than the widget
  QPoint drawPos = m_imageOffset;
  if (scaledSize.width() < width()) {
    drawPos.setX((width() - scaledSize.width()) / 2);
  }
  if (scaledSize.height() < height()) {
    drawPos.setY((height() - scaledSize.height()) / 2);
  }

  QRect drawRect(drawPos, scaledSize);
  painter.drawPixmap(drawRect, m_pixmap);
}

void ImageDisplayWidget::wheelEvent(QWheelEvent *event) {
  if (m_pixmap.isNull())
    return;

  const double factor =
      (event->angleDelta().y() > 0) ? ZOOM_STEP : 1.0 / ZOOM_STEP;

  // Store the position before zoom
  QPoint oldPos = mapToImage(event->position().toPoint());

  setZoomFactor(m_zoomFactor * factor);

  // Adjust offset to keep the same point under the mouse
  QPoint newPos = mapToImage(event->position().toPoint());
  m_imageOffset += (oldPos - newPos);

  update();
  event->accept();
}

void ImageDisplayWidget::mousePressEvent(QMouseEvent *event) {
  if (m_pixmap.isNull())
    return;

  if (event->button() == Qt::LeftButton) {
    if (m_selectionEnabled) {
      m_selecting = true;
      m_selectionStart = event->pos();
      m_selectionRect = QRect(m_selectionStart, QSize());

      if (!m_rubberBand) {
        m_rubberBand = new QRubberBand(QRubberBand::Rectangle, this);
      }
      m_rubberBand->setGeometry(m_selectionRect);
      m_rubberBand->show();
    } else {
      m_dragging = true;
      m_lastPanPoint = event->pos();
      setCursor(Qt::ClosedHandCursor);
    }

    emit mouseClicked(mapToImage(event->pos()));
  }

  event->accept();
}

void ImageDisplayWidget::mouseMoveEvent(QMouseEvent *event) {
  if (m_pixmap.isNull())
    return;

  if (m_selecting && (event->buttons() & Qt::LeftButton)) {
    m_selectionRect = QRect(m_selectionStart, event->pos()).normalized();
    if (m_rubberBand) {
      m_rubberBand->setGeometry(m_selectionRect);
    }
  } else if (m_dragging && (event->buttons() & Qt::LeftButton)) {
    QPoint delta = event->pos() - m_lastPanPoint;
    m_imageOffset += delta;
    m_lastPanPoint = event->pos();
    update();
  }

  emit mouseMoved(mapToImage(event->pos()));
  event->accept();
}

void ImageDisplayWidget::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    if (m_selecting) {
      m_selecting = false;
      if (m_rubberBand) {
        m_rubberBand->hide();
      }

      // Convert selection to image coordinates
      QRect imageSelection;
      if (!m_selectionRect.isEmpty()) {
        QPoint topLeft = mapToImage(m_selectionRect.topLeft());
        QPoint bottomRight = mapToImage(m_selectionRect.bottomRight());
        imageSelection = QRect(topLeft, bottomRight).normalized();
      }

      emit selectionChanged(imageSelection);
    } else if (m_dragging) {
      m_dragging = false;
      setCursor(Qt::ArrowCursor);
    }
  }

  event->accept();
}

void ImageDisplayWidget::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  updateScrollBars();
}

void ImageDisplayWidget::updateDisplay() { update(); }

QPixmap ImageDisplayWidget::matToQPixmap(const cv::Mat &mat) {
  if (mat.empty())
    return QPixmap();

  cv::Mat rgbMat;
  if (mat.channels() == 3) {
    cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
  } else if (mat.channels() == 1) {
    cv::cvtColor(mat, rgbMat, cv::COLOR_GRAY2RGB);
  } else {
    rgbMat = mat.clone();
  }

  QImage qimg(rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step,
              QImage::Format_RGB888);
  return QPixmap::fromImage(qimg);
}

void ImageDisplayWidget::updateScrollBars() {
  // This could be extended if we add scroll bars in the future
}

QPoint ImageDisplayWidget::mapToImage(const QPoint &widgetPos) const {
  if (m_pixmap.isNull())
    return QPoint();

  // Account for zoom and offset
  QPoint adjustedPos = widgetPos - m_imageOffset;

  // Scale back to original image coordinates
  int x = static_cast<int>(adjustedPos.x() / m_zoomFactor);
  int y = static_cast<int>(adjustedPos.y() / m_zoomFactor);

  // Clamp to image bounds
  x = std::clamp(x, 0, m_originalImage.cols - 1);
  y = std::clamp(y, 0, m_originalImage.rows - 1);

  return QPoint(x, y);
}

QPoint ImageDisplayWidget::mapFromImage(const QPoint &imagePos) const {
  if (m_pixmap.isNull())
    return QPoint();

  // Scale to current zoom level
  int x = static_cast<int>(imagePos.x() * m_zoomFactor);
  int y = static_cast<int>(imagePos.y() * m_zoomFactor);

  // Add offset
  return QPoint(x, y) + m_imageOffset;
}

} // namespace stereo_vision::gui
