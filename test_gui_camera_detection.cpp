#include "camera_manager.hpp"
#include <QApplication>
#include <QDebug>
#include <iostream>

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);

  std::cout << "=== GUI Camera Manager Test ===" << std::endl;
  std::cout << "Testing camera detection in Qt application context..."
            << std::endl;

  // Create camera manager in Qt context
  stereo_vision::CameraManager cam_mgr;

  std::cout << "Starting camera detection..." << std::endl;
  int num_cameras = cam_mgr.detectCameras();

  std::cout << "Camera detection result: " << num_cameras << " cameras found"
            << std::endl;

  auto camera_list = cam_mgr.getCameraList();
  std::cout << "Camera list:" << std::endl;
  for (size_t i = 0; i < camera_list.size(); ++i) {
    std::cout << "  " << i << ": " << camera_list[i] << std::endl;
  }

  std::cout << "Test completed." << std::endl;
  return 0;
}
