#include <QApplication>
#include <QMessageBox>
#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);

  // Create a main window instead of just a message box
  QWidget window;
  window.setWindowTitle("Stereo Vision App - Build Test");
  window.resize(400, 300);
  
  QVBoxLayout *layout = new QVBoxLayout(&window);
  
  QLabel *title = new QLabel("Stereo Vision 3D Point Cloud Generator");
  title->setStyleSheet("font-size: 16px; font-weight: bold; color: #2E7D32;");
  layout->addWidget(title);
  
  QLabel *status = new QLabel("✅ Build successful!\n✅ Qt5 GUI working!\n✅ Ready for stereo vision processing");
  status->setStyleSheet("font-size: 12px; color: #333;");
  layout->addWidget(status);
  
  QPushButton *closeBtn = new QPushButton("Close");
  layout->addWidget(closeBtn);
  
  // Connect close button
  QObject::connect(closeBtn, &QPushButton::clicked, &app, &QApplication::quit);
  
  window.show();
  
  return app.exec();
}
