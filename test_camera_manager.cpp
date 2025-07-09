#include "camera_manager.hpp"
#include <iostream>

int main() {
  std::cout << "=== Camera Manager Test ===" << std::endl;

  stereo_vision::CameraManager manager;

  std::cout << "Detecting cameras..." << std::endl;
  int numCameras = manager.detectCameras();

  std::cout << "\nCamera detection results:" << std::endl;
  std::cout << "Number of usable cameras: " << numCameras << std::endl;

  const auto &cameraList = manager.getCameraList();

  if (numCameras > 0) {
    std::cout << "\nAvailable cameras:" << std::endl;
    for (int i = 0; i < numCameras; i++) {
      std::cout << "  " << i << ": " << cameraList[i] << std::endl;
      std::cout << "     Device index: " << manager.getDeviceIndex(i)
                << std::endl;
    }

    // Test opening the first camera if available
    if (numCameras >= 1) {
      std::cout << "\nTesting camera 0..." << std::endl;
      if (manager.openCameras(0, 0)) {
        std::cout << "Camera 0 opened successfully!" << std::endl;

        cv::Mat frame1, frame2;
        if (manager.grabFrames(frame1, frame2)) {
          std::cout << "Frame capture successful: " << frame1.cols << "x"
                    << frame1.rows << std::endl;
        } else {
          std::cout << "Frame capture failed" << std::endl;
        }

        manager.closeCameras();
      } else {
        std::cout << "Failed to open camera 0" << std::endl;
      }
    }
  } else {
    std::cout << "No usable cameras found!" << std::endl;
  }

  return 0;
}
