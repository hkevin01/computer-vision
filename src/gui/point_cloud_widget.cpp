#include "gui/point_cloud_widget.hpp"
#include <QApplication>
#include <QMatrix4x4>
#include <QOpenGLContext>
#include <QVBoxLayout>
#include <QVector3D>
#include <QtMath>
#include <cmath>

namespace stereo_vision::gui {

PointCloudWidget::PointCloudWidget(QWidget *parent)
    : QOpenGLWidget(parent),
      m_pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>),
      m_shaderProgram(nullptr), m_pointCloudDirty(false), m_mousePressed(false),
      m_pointSize(2.0f), m_backgroundColor(Qt::black), m_wireframeMode(false),
      m_showAxes(true), m_showGrid(true), m_animate(false),
      m_cameraDistance(5.0f), m_cameraYaw(45.0f), m_cameraPitch(30.0f),
      m_animationAngle(0.0f), m_animationTimer(new QTimer(this)) {

  setMinimumSize(400, 300);
  setFocusPolicy(Qt::StrongFocus);

  // Initialize camera
  m_cameraPosition = QVector3D(0, 0, m_cameraDistance);
  m_cameraTarget = QVector3D(0, 0, 0);
  m_cameraUp = QVector3D(0, 1, 0);

  // Animation timer
  connect(m_animationTimer, &QTimer::timeout, this, &PointCloudWidget::animate);
  m_animationTimer->setInterval(16); // ~60 FPS
}

PointCloudWidget::~PointCloudWidget() {
  makeCurrent();
  m_vao.release();
  m_vertexBuffer.release();
  delete m_shaderProgram;
  doneCurrent();
}

void PointCloudWidget::setPointCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
  m_pointCloud = cloud;
  m_pointCloudDirty = true;
  updatePointCloudData();
  emit pointCloudChanged();
  update();
}

void PointCloudWidget::clearPointCloud() {
  m_pointCloud->clear();
  m_vertices.clear();
  m_pointCloudDirty = true;
  setupBuffers();
  emit pointCloudChanged();
  update();
}

void PointCloudWidget::resetView() {
  m_cameraDistance = 5.0f;
  m_cameraYaw = 45.0f;
  m_cameraPitch = 30.0f;
  updateCamera();
  emit viewChanged();
  update();
}

void PointCloudWidget::setViewFromTop() {
  m_cameraYaw = 0.0f;
  m_cameraPitch = 90.0f;
  updateCamera();
  emit viewChanged();
  update();
}

void PointCloudWidget::setViewFromSide() {
  m_cameraYaw = 90.0f;
  m_cameraPitch = 0.0f;
  updateCamera();
  emit viewChanged();
  update();
}

void PointCloudWidget::setViewFromFront() {
  m_cameraYaw = 0.0f;
  m_cameraPitch = 0.0f;
  updateCamera();
  emit viewChanged();
  update();
}

void PointCloudWidget::setPointSize(float size) {
  m_pointSize = std::clamp(size, 0.1f, 10.0f);
  update();
}

void PointCloudWidget::setBackgroundColor(const QColor &color) {
  m_backgroundColor = color;
  update();
}

void PointCloudWidget::setWireframeMode(bool enabled) {
  m_wireframeMode = enabled;
  update();
}

void PointCloudWidget::setShowAxes(bool show) {
  m_showAxes = show;
  update();
}

void PointCloudWidget::setShowGrid(bool show) {
  m_showGrid = show;
  update();
}

void PointCloudWidget::exportToImage(const QString &filename) {
  QImage image = grabFramebuffer();
  image.save(filename);
}

void PointCloudWidget::initializeGL() {
  initializeOpenGLFunctions();

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);

  setupShaders();
  setupBuffers();
}

void PointCloudWidget::paintGL() {
  // Clear buffers
  glClearColor(m_backgroundColor.redF(), m_backgroundColor.greenF(),
               m_backgroundColor.blueF(), m_backgroundColor.alphaF());
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (!m_shaderProgram || !m_shaderProgram->bind()) {
    return;
  }

  // Update matrices
  QMatrix4x4 mvp = m_projection * m_view * m_model;
  m_shaderProgram->setUniformValue("mvp", mvp);
  m_shaderProgram->setUniformValue("pointSize", m_pointSize);

  // Draw point cloud
  if (!m_vertices.empty()) {
    m_vao.bind();
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_vertices.size()));
    m_vao.release();
  }

  // Draw additional elements
  if (m_showAxes) {
    drawAxes();
  }

  if (m_showGrid) {
    drawGrid();
  }

  m_shaderProgram->release();
}

void PointCloudWidget::resizeGL(int width, int height) {
  glViewport(0, 0, width, height);

  float aspect = static_cast<float>(width) / static_cast<float>(height);
  m_projection.setToIdentity();
  m_projection.perspective(45.0f, aspect, 0.1f, 100.0f);
}

void PointCloudWidget::mousePressEvent(QMouseEvent *event) {
  m_mousePressed = true;
  m_lastMousePos = event->pos();
  m_mouseButton = event->button();
}

void PointCloudWidget::mouseMoveEvent(QMouseEvent *event) {
  if (!m_mousePressed)
    return;

  QPoint delta = event->pos() - m_lastMousePos;
  m_lastMousePos = event->pos();

  if (m_mouseButton == Qt::LeftButton) {
    // Rotate camera
    m_cameraYaw += delta.x() * MOUSE_SENSITIVITY;
    m_cameraPitch -= delta.y() * MOUSE_SENSITIVITY;
    m_cameraPitch = std::clamp(m_cameraPitch, -89.0f, 89.0f);
    updateCamera();
  } else if (m_mouseButton == Qt::RightButton) {
    // Pan camera
    float sensitivity = 0.01f;
    QVector3D right =
        QVector3D::crossProduct(m_cameraTarget - m_cameraPosition, m_cameraUp)
            .normalized();
    QVector3D up =
        QVector3D::crossProduct(right, m_cameraTarget - m_cameraPosition)
            .normalized();

    m_cameraTarget += right * delta.x() * sensitivity;
    m_cameraTarget += up * delta.y() * sensitivity;
    updateCamera();
  }

  emit viewChanged();
  update();
}

void PointCloudWidget::mouseReleaseEvent(QMouseEvent *event) {
  Q_UNUSED(event)
  m_mousePressed = false;
}

void PointCloudWidget::wheelEvent(QWheelEvent *event) {
  float delta = event->angleDelta().y() / 120.0f;
  m_cameraDistance *= (1.0f - delta * 0.1f);
  m_cameraDistance =
      std::clamp(m_cameraDistance, MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE);
  updateCamera();
  emit viewChanged();
  update();
}

void PointCloudWidget::keyPressEvent(QKeyEvent *event) {
  switch (event->key()) {
  case Qt::Key_R:
    resetView();
    break;
  case Qt::Key_1:
    setViewFromFront();
    break;
  case Qt::Key_2:
    setViewFromSide();
    break;
  case Qt::Key_3:
    setViewFromTop();
    break;
  case Qt::Key_A:
    m_animate = !m_animate;
    if (m_animate) {
      m_animationTimer->start();
    } else {
      m_animationTimer->stop();
    }
    break;
  case Qt::Key_G:
    setShowGrid(!m_showGrid);
    break;
  case Qt::Key_X:
    setShowAxes(!m_showAxes);
    break;
  default:
    QOpenGLWidget::keyPressEvent(event);
  }
}

void PointCloudWidget::animate() {
  m_animationAngle += 1.0f;
  if (m_animationAngle >= 360.0f) {
    m_animationAngle = 0.0f;
  }

  m_cameraYaw = m_animationAngle;
  updateCamera();
  update();
}

void PointCloudWidget::setupShaders() {
  delete m_shaderProgram;
  m_shaderProgram = new QOpenGLShaderProgram(this);

  // Vertex shader
  const char *vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec3 color;
    
    uniform mat4 mvp;
    uniform float pointSize;
    
    out vec3 fragColor;
    
    void main() {
      gl_Position = mvp * vec4(position, 1.0);
      gl_PointSize = pointSize;
      fragColor = color;
    }
  )";

  // Fragment shader
  const char *fragmentShaderSource = R"(
    #version 330 core
    in vec3 fragColor;
    out vec4 color;
    
    void main() {
      color = vec4(fragColor, 1.0);
    }
  )";

  m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                           vertexShaderSource);
  m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                           fragmentShaderSource);
  m_shaderProgram->link();
}

void PointCloudWidget::setupBuffers() {
  if (!m_vao.isCreated()) {
    m_vao.create();
  }

  if (!m_vertexBuffer.isCreated()) {
    m_vertexBuffer.create();
  }

  m_vao.bind();
  m_vertexBuffer.bind();

  if (!m_vertices.empty()) {
    m_vertexBuffer.allocate(
        m_vertices.data(),
        static_cast<int>(m_vertices.size() * sizeof(Vertex)));

    // Position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void *>(offsetof(Vertex, position)));

    // Color attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void *>(offsetof(Vertex, color)));
  }

  m_vao.release();
  m_vertexBuffer.release();
}

void PointCloudWidget::updatePointCloudData() {
  if (!m_pointCloud || m_pointCloud->empty()) {
    m_vertices.clear();
    setupBuffers();
    return;
  }

  m_vertices.clear();
  m_vertices.reserve(m_pointCloud->size());

  for (const auto &point : m_pointCloud->points) {
    if (std::isfinite(point.x) && std::isfinite(point.y) &&
        std::isfinite(point.z)) {
      Vertex vertex;
      vertex.position = QVector3D(point.x, point.y, point.z);
      vertex.color =
          QVector3D(point.r / 255.0f, point.g / 255.0f, point.b / 255.0f);
      m_vertices.push_back(vertex);
    }
  }

  setupBuffers();
}

void PointCloudWidget::updateCamera() {
  // Convert spherical coordinates to cartesian
  float yawRad = qDegreesToRadians(m_cameraYaw);
  float pitchRad = qDegreesToRadians(m_cameraPitch);

  QVector3D direction;
  direction.setX(cos(pitchRad) * cos(yawRad));
  direction.setY(sin(pitchRad));
  direction.setZ(cos(pitchRad) * sin(yawRad));

  m_cameraPosition = m_cameraTarget - direction * m_cameraDistance;

  m_view.setToIdentity();
  m_view.lookAt(m_cameraPosition, m_cameraTarget, m_cameraUp);

  m_model.setToIdentity();
}

void PointCloudWidget::drawAxes() {
  // This would require additional shader setup for lines
  // Simplified implementation - could be expanded
}

void PointCloudWidget::drawGrid() {
  // This would require additional shader setup for lines
  // Simplified implementation - could be expanded
}

} // namespace stereo_vision::gui
