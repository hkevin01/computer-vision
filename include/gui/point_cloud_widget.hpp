#pragma once

#include <QtWidgets/QWidget>
#include <QtOpenGL/QOpenGLWidget>
#include <QtOpenGL/QOpenGLFunctions>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtGui/QMatrix4x4>
#include <QtCore/QTimer>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

class PointCloudWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit PointCloudWidget(QWidget *parent = nullptr);
    ~PointCloudWidget();

    void setPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
    void clearPointCloud();
    
    // Camera controls
    void resetView();
    void setViewFromTop();
    void setViewFromSide();
    void setViewFromFront();
    
    // Rendering options
    void setPointSize(float size);
    void setBackgroundColor(const QColor &color);
    void setWireframeMode(bool enabled);
    void setShowAxes(bool show);
    void setShowGrid(bool show);
    
    // Export functionality
    void exportToImage(const QString &filename);

signals:
    void pointCloudChanged();
    void viewChanged();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void animate();

private:
    struct Vertex {
        QVector3D position;
        QVector3D color;
    };
    
    void setupShaders();
    void setupBuffers();
    void updatePointCloudData();
    void drawAxes();
    void drawGrid();
    
    // Matrices
    QMatrix4x4 m_projection;
    QMatrix4x4 m_view;
    QMatrix4x4 m_model;
    
    // Camera parameters
    QVector3D m_cameraPosition;
    QVector3D m_cameraTarget;
    QVector3D m_cameraUp;
    float m_cameraDistance;
    float m_cameraYaw;
    float m_cameraPitch;
    
    // Mouse interaction
    bool m_mousePressed;
    QPoint m_lastMousePos;
    Qt::MouseButton m_mouseButton;
    
    // Rendering options
    float m_pointSize;
    QColor m_backgroundColor;
    bool m_wireframeMode;
    bool m_showAxes;
    bool m_showGrid;
    bool m_animate;
    
    // OpenGL objects
    QOpenGLShaderProgram *m_shaderProgram;
    QOpenGLBuffer m_vertexBuffer;
    QOpenGLVertexArrayObject m_vao;
    
    // Point cloud data
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_pointCloud;
    std::vector<Vertex> m_vertices;
    bool m_pointCloudDirty;
    
    // Animation
    QTimer *m_animationTimer;
    float m_animationAngle;
    
    // Constants
    static constexpr float MIN_CAMERA_DISTANCE = 0.1f;
    static constexpr float MAX_CAMERA_DISTANCE = 100.0f;
    static constexpr float CAMERA_SPEED = 0.1f;
    static constexpr float MOUSE_SENSITIVITY = 0.5f;
};
