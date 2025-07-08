#include "gui/point_cloud_widget.hpp"
#include <QtCore/QDebug>

PointCloudWidget::PointCloudWidget(QWidget *parent)
    : QOpenGLWidget(parent), m_cameraPosition(0.0f, 0.0f, 5.0f), m_cameraTarget(0.0f, 0.0f, 0.0f), m_cameraUp(0.0f, 1.0f, 0.0f), m_cameraDistance(5.0f), m_cameraYaw(0.0f), m_cameraPitch(0.0f), m_mousePressed(false), m_pointSize(1.0f), m_backgroundColor(Qt::black), m_wireframeMode(false), m_showAxes(true), m_showGrid(true), m_animate(false), m_shaderProgram(nullptr), m_pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>()), m_pointCloudDirty(false), m_animationTimer(new QTimer(this)), m_animationAngle(0.0f)
{
    setFocusPolicy(Qt::StrongFocus);

    connect(m_animationTimer, &QTimer::timeout, this, &PointCloudWidget::animate);
    m_animationTimer->start(16); // ~60 FPS
}

PointCloudWidget::~PointCloudWidget()
{
    makeCurrent();
    delete m_shaderProgram;
    doneCurrent();
}

void PointCloudWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(m_backgroundColor.redF(), m_backgroundColor.greenF(),
                 m_backgroundColor.blueF(), m_backgroundColor.alphaF());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    setupShaders();
    setupBuffers();
}

void PointCloudWidget::setupShaders()
{
    m_shaderProgram = new QOpenGLShaderProgram();

    // Vertex shader
    const char *vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 color;
        
        uniform mat4 mvp;
        uniform float pointSize;
        
        out vec3 fragColor;
        
        void main()
        {
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
        
        void main()
        {
            color = vec4(fragColor, 1.0);
        }
    )";

    if (!m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource))
    {
        qWarning() << "Failed to compile vertex shader:" << m_shaderProgram->log();
    }

    if (!m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource))
    {
        qWarning() << "Failed to compile fragment shader:" << m_shaderProgram->log();
    }

    if (!m_shaderProgram->link())
    {
        qWarning() << "Failed to link shader program:" << m_shaderProgram->log();
    }
}

void PointCloudWidget::setupBuffers()
{
    m_vao.create();
    m_vao.bind();

    m_vertexBuffer.create();
    m_vertexBuffer.bind();

    m_vertexBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);

    // Enable vertex attributes
    m_shaderProgram->enableAttributeArray(0);
    m_shaderProgram->setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(Vertex));

    m_shaderProgram->enableAttributeArray(1);
    m_shaderProgram->setAttributeBuffer(1, GL_FLOAT, sizeof(QVector3D), 3, sizeof(Vertex));

    m_vao.release();
}

void PointCloudWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!m_shaderProgram || !m_shaderProgram->bind())
    {
        return;
    }

    // Calculate camera position
    float x = m_cameraDistance * cos(qDegreesToRadians(m_cameraPitch)) * cos(qDegreesToRadians(m_cameraYaw));
    float y = m_cameraDistance * sin(qDegreesToRadians(m_cameraPitch));
    float z = m_cameraDistance * cos(qDegreesToRadians(m_cameraPitch)) * sin(qDegreesToRadians(m_cameraYaw));

    m_cameraPosition = QVector3D(x, y, z);

    // Update view matrix
    m_view.setToIdentity();
    m_view.lookAt(m_cameraPosition, m_cameraTarget, m_cameraUp);

    // Update model matrix
    m_model.setToIdentity();
    if (m_animate)
    {
        m_model.rotate(m_animationAngle, 0, 1, 0);
    }

    // Calculate MVP matrix
    QMatrix4x4 mvp = m_projection * m_view * m_model;

    // Set uniforms
    m_shaderProgram->setUniformValue("mvp", mvp);
    m_shaderProgram->setUniformValue("pointSize", m_pointSize);

    // Update point cloud data if needed
    if (m_pointCloudDirty)
    {
        updatePointCloudData();
        m_pointCloudDirty = false;
    }

    // Draw point cloud
    if (!m_vertices.empty())
    {
        m_vao.bind();
        glDrawArrays(GL_POINTS, 0, m_vertices.size());
        m_vao.release();
    }

    // Draw axes and grid if enabled
    if (m_showAxes)
    {
        drawAxes();
    }

    if (m_showGrid)
    {
        drawGrid();
    }

    m_shaderProgram->release();
}

void PointCloudWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);

    m_projection.setToIdentity();
    m_projection.perspective(45.0f, float(width) / float(height), 0.1f, 100.0f);
}

void PointCloudWidget::setPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    m_pointCloud = cloud;
    m_pointCloudDirty = true;
    update();
    emit pointCloudChanged();
}

void PointCloudWidget::clearPointCloud()
{
    m_pointCloud->clear();
    m_vertices.clear();
    m_pointCloudDirty = true;
    update();
    emit pointCloudChanged();
}

void PointCloudWidget::updatePointCloudData()
{
    m_vertices.clear();

    if (!m_pointCloud || m_pointCloud->empty())
    {
        return;
    }

    m_vertices.reserve(m_pointCloud->size());

    for (const auto &point : m_pointCloud->points)
    {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
        {
            Vertex vertex;
            vertex.position = QVector3D(point.x, point.y, point.z);
            vertex.color = QVector3D(point.r / 255.0f, point.g / 255.0f, point.b / 255.0f);
            m_vertices.push_back(vertex);
        }
    }

    // Update buffer
    m_vao.bind();
    m_vertexBuffer.bind();
    m_vertexBuffer.allocate(m_vertices.data(), m_vertices.size() * sizeof(Vertex));
    m_vao.release();
}

void PointCloudWidget::mousePressEvent(QMouseEvent *event)
{
    m_mousePressed = true;
    m_lastMousePos = event->pos();
    m_mouseButton = event->button();
}

void PointCloudWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (!m_mousePressed)
        return;

    QPoint delta = event->pos() - m_lastMousePos;

    if (m_mouseButton == Qt::LeftButton)
    {
        // Rotate camera
        m_cameraYaw += delta.x() * MOUSE_SENSITIVITY;
        m_cameraPitch -= delta.y() * MOUSE_SENSITIVITY;

        // Clamp pitch
        m_cameraPitch = qBound(-89.0f, m_cameraPitch, 89.0f);

        update();
        emit viewChanged();
    }
    else if (m_mouseButton == Qt::RightButton)
    {
        // Pan camera
        QVector3D right = QVector3D::crossProduct(m_cameraTarget - m_cameraPosition, m_cameraUp).normalized();
        QVector3D up = QVector3D::crossProduct(right, m_cameraTarget - m_cameraPosition).normalized();

        float sensitivity = m_cameraDistance * 0.001f;
        m_cameraTarget += (right * delta.x() - up * delta.y()) * sensitivity;

        update();
        emit viewChanged();
    }

    m_lastMousePos = event->pos();
}

void PointCloudWidget::mouseReleaseEvent(QMouseEvent *event)
{
    Q_UNUSED(event)
    m_mousePressed = false;
}

void PointCloudWidget::wheelEvent(QWheelEvent *event)
{
    const float delta = event->angleDelta().y() / 1200.0f;
    m_cameraDistance -= delta * m_cameraDistance * 0.1f;
    m_cameraDistance = qBound(MIN_CAMERA_DISTANCE, m_cameraDistance, MAX_CAMERA_DISTANCE);

    update();
    emit viewChanged();
}

void PointCloudWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_R:
        resetView();
        break;
    case Qt::Key_A:
        m_animate = !m_animate;
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

void PointCloudWidget::resetView()
{
    m_cameraDistance = 5.0f;
    m_cameraYaw = 0.0f;
    m_cameraPitch = 0.0f;
    m_cameraTarget = QVector3D(0.0f, 0.0f, 0.0f);

    update();
    emit viewChanged();
}

void PointCloudWidget::setViewFromTop()
{
    m_cameraYaw = 0.0f;
    m_cameraPitch = 90.0f;
    update();
    emit viewChanged();
}

void PointCloudWidget::setViewFromSide()
{
    m_cameraYaw = 90.0f;
    m_cameraPitch = 0.0f;
    update();
    emit viewChanged();
}

void PointCloudWidget::setViewFromFront()
{
    m_cameraYaw = 0.0f;
    m_cameraPitch = 0.0f;
    update();
    emit viewChanged();
}

void PointCloudWidget::setPointSize(float size)
{
    m_pointSize = qBound(0.1f, size, 10.0f);
    update();
}

void PointCloudWidget::setBackgroundColor(const QColor &color)
{
    m_backgroundColor = color;
    makeCurrent();
    glClearColor(color.redF(), color.greenF(), color.blueF(), color.alphaF());
    doneCurrent();
    update();
}

void PointCloudWidget::setWireframeMode(bool enabled)
{
    m_wireframeMode = enabled;
    update();
}

void PointCloudWidget::setShowAxes(bool show)
{
    m_showAxes = show;
    update();
}

void PointCloudWidget::setShowGrid(bool show)
{
    m_showGrid = show;
    update();
}

void PointCloudWidget::animate()
{
    if (m_animate)
    {
        m_animationAngle += 1.0f;
        if (m_animationAngle >= 360.0f)
        {
            m_animationAngle = 0.0f;
        }
        update();
    }
}

void PointCloudWidget::drawAxes()
{
    // TODO: Implement axes drawing using OpenGL
    // This would require a separate shader and vertex buffer for lines
}

void PointCloudWidget::drawGrid()
{
    // TODO: Implement grid drawing using OpenGL
    // This would require a separate shader and vertex buffer for lines
}

void PointCloudWidget::exportToImage(const QString &filename)
{
    QImage image = grabFramebuffer();
    image.save(filename);
}

#include "point_cloud_widget.moc"
