#include "gui/main_window.hpp"
#include "gui/modern_theme.hpp"
#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QIcon>
#include <QOpenGLContext>
#include <QStandardPaths>
#include <QString>
#include <QStyleFactory>

using namespace stereo_vision::gui;

int main(int argc, char *argv[]) {
  // Enable performance optimizations before creating QApplication
  PerformanceOptimizer::enableHighDPI();
  PerformanceOptimizer::enableGPUAcceleration();

  QApplication app(argc, argv);

  // Apply modern Windows 11 theme and performance optimizations
  ModernTheme::applyTheme(&app);
  PerformanceOptimizer::optimizeQtPerformance();
  PerformanceOptimizer::optimizeOpenGL();

  // Set application properties
  app.setApplicationName("Stereo Vision 3D Point Cloud Generator");
  app.setApplicationVersion("2.0.0");
  app.setOrganizationName("Computer Vision Studio");
  app.setApplicationDisplayName("Stereo Vision Pro");

  // Set application icon and window icon
  app.setWindowIcon(QIcon(":/icons/app_icon.png"));

  // Create application data directory
  QString dataDir =
      QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
  QDir().mkpath(dataDir);

  // Check for help argument
  for (int i = 1; i < argc; ++i) {
    QString arg = QString::fromLocal8Bit(argv[i]);
    if (arg == "--help" || arg == "-h") {
      qDebug() << "Stereo Vision 3D Point Cloud Generator v2.0";
      qDebug()
          << "Modern C++ application for real-time stereo vision processing";
      qDebug() << "";
      qDebug() << "Usage: stereo_vision_app [options]";
      qDebug() << "Options:";
      qDebug() << "  --help             Show this help";
      qDebug() << "  --version          Show version information";
      qDebug() << "  --gpu-info         Show GPU acceleration status";
      qDebug() << "";
      qDebug() << "Features:";
      qDebug() << "  • Camera calibration wizard";
      qDebug() << "  • Real-time stereo processing";
      qDebug() << "  • 3D point cloud generation";
      qDebug() << "  • GPU acceleration (CUDA/HIP)";
      qDebug() << "  • Windows 11 modern UI";
      return 0;
    }
    if (arg == "--version") {
      qDebug() << "Stereo Vision Pro v2.0.0";
      qDebug() << "Built with Qt" << QT_VERSION_STR;
      qDebug() << "OpenCV support enabled";
      qDebug() << "PCL integration enabled";
      return 0;
    }
    if (arg == "--gpu-info") {
      qDebug() << "GPU Acceleration Status:";
      qDebug() << "OpenGL:"
               << (QOpenGLContext::openGLModuleType() == QOpenGLContext::LibGL
                       ? "Desktop"
                       : "ES");
      return 0;
    }
  }

  try {
    MainWindow window;

    // Apply modern styling and animations
    window.setGraphicsEffect(ModernTheme::createDropShadow());

    // Show with fade-in animation
    auto *fadeIn = ModernTheme::createFadeAnimation(&window);
    fadeIn->setStartValue(0.0);
    fadeIn->setEndValue(1.0);

    window.show();
    fadeIn->start();

    return app.exec();

  } catch (const std::exception &e) {
    qCritical() << "Application error:" << e.what();
    return 1;
  }
}
