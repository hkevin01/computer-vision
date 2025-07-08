#include "gui/image_display_widget.hpp"
#include <QtCore/QDebug>

ImageDisplayWidget::ImageDisplayWidget(QWidget *parent)
    : QWidget(parent), m_zoomFactor(1.0), m_imageOffset(0, 0), m_dragging(false), m_selectionEnabled(false), m_selecting(false), m_rubberBand(new QRubberBand(QRubberBand::Rectangle, this))
{
    setMinimumSize(320, 240);
    setBackgroundRole(QPalette::Dark);
    setAutoFillBackground(true);

    // Enable mouse tracking for pan functionality
    setMouseTracking(true);

    m_rubberBand->hide();
}

ImageDisplayWidget::~ImageDisplayWidget()
{
    delete m_rubberBand;
}

void ImageDisplayWidget::setImage(const cv::Mat &image)
{
    m_originalImage = image.clone();
    m_imagePath.clear();

    if (!image.empty())
    {
        m_pixmap = matToQPixmap(image);
        zoomToFit();
    }
    else
    {
        m_pixmap = QPixmap();
        m_zoomFactor = 1.0;
        m_imageOffset = QPoint(0, 0);
    }

    update();
    emit imageChanged();
}

void ImageDisplayWidget::setImage(const QString &imagePath)
{
    cv::Mat image = cv::imread(imagePath.toStdString());
    m_imagePath = imagePath;
    setImage(image);
}

void ImageDisplayWidget::clearImage()
{
    m_originalImage = cv::Mat();
    m_pixmap = QPixmap();
    m_imagePath.clear();
    m_zoomFactor = 1.0;
    m_imageOffset = QPoint(0, 0);
    clearSelection();

    update();
    emit imageChanged();
}

QPixmap ImageDisplayWidget::matToQPixmap(const cv::Mat &mat)
{
    switch (mat.type())
    {
    case CV_8UC4:
    {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return QPixmap::fromImage(qimg);
    }
    case CV_8UC3:
    {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return QPixmap::fromImage(qimg.rgbSwapped());
    }
    case CV_8UC1:
    {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return QPixmap::fromImage(qimg);
    }
    default:
        qWarning("ImageDisplayWidget::matToQPixmap() - cv::Mat format not supported");
        return QPixmap();
    }
}

void ImageDisplayWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    if (m_pixmap.isNull())
    {
        painter.fillRect(rect(), Qt::darkGray);
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No Image");
        return;
    }

    // Calculate scaled pixmap size
    QSize scaledSize = m_pixmap.size() * m_zoomFactor;

    // Calculate position to center the image
    QPoint position = rect().center() - QPoint(scaledSize.width() / 2, scaledSize.height() / 2) + m_imageOffset;

    // Draw the scaled image
    QRect drawRect(position, scaledSize);
    painter.drawPixmap(drawRect, m_pixmap);

    // Draw selection rectangle if active
    if (m_selecting && !m_selectionRect.isEmpty())
    {
        painter.setPen(QPen(Qt::red, 2, Qt::DashLine));
        painter.drawRect(m_selectionRect);
    }
}

void ImageDisplayWidget::zoomIn()
{
    setZoomFactor(m_zoomFactor * ZOOM_STEP);
}

void ImageDisplayWidget::zoomOut()
{
    setZoomFactor(m_zoomFactor / ZOOM_STEP);
}

void ImageDisplayWidget::zoomToFit()
{
    if (m_pixmap.isNull())
        return;

    QSize imageSize = m_pixmap.size();
    QSize widgetSize = size();

    double scaleX = static_cast<double>(widgetSize.width()) / imageSize.width();
    double scaleY = static_cast<double>(widgetSize.height()) / imageSize.height();

    setZoomFactor(std::min(scaleX, scaleY) * 0.9); // 90% to leave some margin
    m_imageOffset = QPoint(0, 0);
    update();
}

void ImageDisplayWidget::zoomToActualSize()
{
    setZoomFactor(1.0);
    m_imageOffset = QPoint(0, 0);
    update();
}

void ImageDisplayWidget::resetZoom()
{
    zoomToActualSize();
}

void ImageDisplayWidget::setZoomFactor(double factor)
{
    factor = std::max(MIN_ZOOM, std::min(MAX_ZOOM, factor));

    if (std::abs(factor - m_zoomFactor) > 1e-6)
    {
        m_zoomFactor = factor;
        update();
        emit zoomChanged(factor);
    }
}

void ImageDisplayWidget::wheelEvent(QWheelEvent *event)
{
    if (m_pixmap.isNull())
        return;

    const double numDegrees = event->angleDelta().y() / 8.0;
    const double numSteps = numDegrees / 15.0;

    if (numSteps > 0)
    {
        zoomIn();
    }
    else
    {
        zoomOut();
    }

    event->accept();
}

void ImageDisplayWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastPanPoint = event->pos();

    if (event->button() == Qt::LeftButton)
    {
        if (m_selectionEnabled)
        {
            m_selecting = true;
            m_selectionStart = event->pos();
            m_selectionRect = QRect(m_selectionStart, QSize());
            m_rubberBand->setGeometry(m_selectionRect);
            m_rubberBand->show();
        }
        else
        {
            m_dragging = true;
            setCursor(Qt::ClosedHandCursor);
        }
    }

    emit mouseClicked(event->pos());
}

void ImageDisplayWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (m_selecting && (event->buttons() & Qt::LeftButton))
    {
        m_selectionRect = QRect(m_selectionStart, event->pos()).normalized();
        m_rubberBand->setGeometry(m_selectionRect);
        emit selectionChanged(m_selectionRect);
    }
    else if (m_dragging && (event->buttons() & Qt::LeftButton))
    {
        QPoint delta = event->pos() - m_lastPanPoint;
        m_imageOffset += delta;
        m_lastPanPoint = event->pos();
        update();
    }

    emit mouseMoved(event->pos());
}

void ImageDisplayWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        if (m_selecting)
        {
            m_selecting = false;
            m_rubberBand->hide();
            emit selectionChanged(m_selectionRect);
        }
        else if (m_dragging)
        {
            m_dragging = false;
            setCursor(Qt::ArrowCursor);
        }
    }
}

void ImageDisplayWidget::clearSelection()
{
    m_selectionRect = QRect();
    m_selecting = false;
    m_rubberBand->hide();
    update();
    emit selectionChanged(m_selectionRect);
}

void ImageDisplayWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    // Could implement auto-fit on resize here if desired
}

void ImageDisplayWidget::updateDisplay()
{
    update();
}

QPoint ImageDisplayWidget::mapToImage(const QPoint &widgetPos) const
{
    if (m_pixmap.isNull())
        return QPoint();

    QSize scaledSize = m_pixmap.size() * m_zoomFactor;
    QPoint imageTopLeft = rect().center() - QPoint(scaledSize.width() / 2, scaledSize.height() / 2) + m_imageOffset;

    QPoint relativePos = widgetPos - imageTopLeft;
    return QPoint(relativePos.x() / m_zoomFactor, relativePos.y() / m_zoomFactor);
}

QPoint ImageDisplayWidget::mapFromImage(const QPoint &imagePos) const
{
    if (m_pixmap.isNull())
        return QPoint();

    QSize scaledSize = m_pixmap.size() * m_zoomFactor;
    QPoint imageTopLeft = rect().center() - QPoint(scaledSize.width() / 2, scaledSize.height() / 2) + m_imageOffset;

    return imageTopLeft + QPoint(imagePos.x() * m_zoomFactor, imagePos.y() * m_zoomFactor);
}

#include "image_display_widget.moc"
