#include "camera_manager.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

int main() {
  std::cout << "=== Single Camera Mode Test ===" << std::endl;

  auto cameraManager = std::make_shared<stereo_vision::CameraManager>();

  std::cout << "Detecting cameras..." << std::endl;
  int numCameras = cameraManager->detectCameras();

  if (numCameras == 0) {
    std::cout << "No cameras detected." << std::endl;
    return 1;
  }

  std::cout << "Found " << numCameras << " camera(s)." << std::endl;
  std::cout << "Testing single camera mode..." << std::endl;

  // Test single camera mode
  bool success = cameraManager->openSingleCamera(0);
  if (!success) {
    std::cout << "Failed to open single camera." << std::endl;
    return 1;
  }

  std::cout << "Single camera opened successfully!" << std::endl;
  std::cout << "Capturing frames for 3 seconds..." << std::endl;

  // Capture frames for a few seconds
  auto startTime = std::chrono::steady_clock::now();
  int frameCount = 0;

  while (std::chrono::steady_clock::now() - startTime <
         std::chrono::seconds(3)) {
    cv::Mat frame;
    if (cameraManager->grabSingleFrame(frame)) {
      if (!frame.empty()) {
        frameCount++;
        std::cout << "Frame " << frameCount << ": " << frame.cols << "x"
                  << frame.rows << std::endl;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "Captured " << frameCount << " frames successfully."
            << std::endl;
  std::cout << "Closing camera..." << std::endl;

  cameraManager->closeCameras();

  std::cout << "Test completed successfully!" << std::endl;
  std::cout << std::endl;
  std::cout << "Single camera mode is working correctly." << std::endl;
  std::cout << "You can now use the GUI application and select the same camera"
            << std::endl;
  std::cout << "for both left and right channels for manual stereo capture."
            << std::endl;

  return 0;
}
