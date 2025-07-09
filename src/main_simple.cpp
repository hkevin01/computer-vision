#include <QApplication>
#include <QMessageBox>

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);

  // Test basic Qt functionality
  QMessageBox::information(
      nullptr, "Stereo Vision App",
      "Stereo Vision 3D Point Cloud Generator\nBasic build successful!");

  return 0;
}
