#include "gui/main_window.hpp"
#include <QApplication>
#include <QDebug>
#include <QString>

using namespace stereo_vision::gui;

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  app.setApplicationName("Stereo Vision 3D Point Cloud Generator");
  app.setApplicationVersion("1.0.0");
  app.setOrganizationName("Computer Vision Project");

  // Check for help argument
  for (int i = 1; i < argc; ++i) {
    QString arg = QString::fromLocal8Bit(argv[i]);
    if (arg == "--help" || arg == "-h") {
      qDebug() << "Stereo Vision 3D Point Cloud Generator";
      qDebug() << "Usage: stereo_vision_app [options]";
      qDebug() << "Options:";
      qDebug() << "  --help             Show this help";
      qDebug() << "  (GUI mode only for now)";
      return 0;
    }
  }

  MainWindow window;
  window.show();

  return app.exec();
}
