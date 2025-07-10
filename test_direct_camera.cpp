#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int main() {
  std::cout << "=== Direct Camera Test ===" << std::endl;
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;
  std::cout << std::endl;

  // Test direct access to cameras
  std::cout << "Testing direct camera access..." << std::endl;

  for (int i = 0; i < 4; i++) {
    std::cout << "Testing camera index " << i << "..." << std::endl;

    // Try V4L2 backend first
    cv::VideoCapture cap(i, cv::CAP_V4L2);
    if (!cap.isOpened()) {
      std::cout << "  V4L2 failed, trying CAP_ANY..." << std::endl;
      cap.open(i, cv::CAP_ANY);
    }

    if (cap.isOpened()) {
      std::cout << "  Camera " << i << " opened successfully!" << std::endl;

      // Try to capture a frame
      cv::Mat frame;
      if (cap.read(frame) && !frame.empty()) {
        double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);

        std::cout << "  Resolution: " << width << "x" << height << std::endl;
        std::cout << "  FPS: " << fps << std::endl;
        std::cout << "  Frame size: " << frame.size() << std::endl;
        std::cout << "  Frame type: " << frame.type() << std::endl;
        std::cout << "  ✅ Camera " << i << " is working!" << std::endl;
      } else {
        std::cout << "  ❌ Camera " << i << " opened but cannot capture frames"
                  << std::endl;
      }

      cap.release();
    } else {
      std::cout << "  ❌ Camera " << i << " failed to open" << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
