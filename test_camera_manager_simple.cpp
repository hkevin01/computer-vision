#include "camera_manager.hpp"
#include <iostream>

int main() {
  std::cout << "=== Camera Manager Test ===" << std::endl;

  stereo_vision::CameraManager cam_mgr;

  std::cout << "Starting camera detection..." << std::endl;
  int num_cameras = cam_mgr.detectCameras();

  std::cout << std::endl;
  std::cout << "Camera detection result: " << num_cameras << " cameras found"
            << std::endl;

  auto camera_list = cam_mgr.getCameraList();
  std::cout << "Camera list:" << std::endl;
  for (size_t i = 0; i < camera_list.size(); ++i) {
    std::cout << "  " << i << ": " << camera_list[i] << std::endl;
  }

  if (num_cameras == 1) {
    std::cout << std::endl;
    std::cout << "Testing single camera mode..." << std::endl;
    if (cam_mgr.openSingleCamera(0)) {
      std::cout << "✅ Single camera opened successfully!" << std::endl;

      // Try to capture a frame
      cv::Mat frame;
      if (cam_mgr.grabSingleFrame(frame)) {
        std::cout << "✅ Frame captured: " << frame.size()
                  << " Type: " << frame.type() << std::endl;
      } else {
        std::cout << "❌ Failed to capture frame" << std::endl;
      }

      cam_mgr.closeCameras();
    } else {
      std::cout << "❌ Failed to open single camera" << std::endl;
    }
  } else if (num_cameras >= 2) {
    std::cout << std::endl;
    std::cout << "Testing stereo camera mode..." << std::endl;
    if (cam_mgr.openCameras(0, 1)) {
      std::cout << "✅ Stereo cameras opened successfully!" << std::endl;

      // Try to capture stereo frames
      cv::Mat left_frame, right_frame;
      if (cam_mgr.grabFrames(left_frame, right_frame)) {
        std::cout << "✅ Stereo frames captured!" << std::endl;
        std::cout << "  Left: " << left_frame.size() << std::endl;
        std::cout << "  Right: " << right_frame.size() << std::endl;
      } else {
        std::cout << "❌ Failed to capture stereo frames" << std::endl;
      }

      cam_mgr.closeCameras();
    } else {
      std::cout << "❌ Failed to open stereo cameras" << std::endl;
    }
  }

  std::cout << std::endl;
  std::cout << "Test completed." << std::endl;
  return 0;
}
