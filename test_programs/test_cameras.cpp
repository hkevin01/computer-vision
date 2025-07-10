#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  std::cout << "OpenCV Camera Detection Test" << std::endl;
  std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
  std::cout << "=============================" << std::endl;

  int count = 0;
  int max_test = 10; // Test up to 10 camera indices

  for (int i = 0; i < max_test; i++) {
    std::cout << "Testing camera index " << i << "... ";
    cv::VideoCapture cap;

    // Try different backends
    bool opened = false;

    // Try V4L2 backend first (Linux)
    if (cap.open(i, cv::CAP_V4L2)) {
      opened = true;
      std::cout << "SUCCESS (V4L2) ";
    }
    // Try default backend
    else if (cap.open(i, cv::CAP_ANY)) {
      opened = true;
      std::cout << "SUCCESS (ANY) ";
    }

    if (opened) {
      count++;

      // Get camera properties
      double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
      double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
      double fps = cap.get(cv::CAP_PROP_FPS);

      std::cout << "- Resolution: " << width << "x" << height;
      std::cout << ", FPS: " << fps << std::endl;

      // Try to capture a frame
      cv::Mat frame;
      if (cap.read(frame)) {
        std::cout << "  Frame capture: SUCCESS (" << frame.cols << "x"
                  << frame.rows << ")" << std::endl;
      } else {
        std::cout << "  Frame capture: FAILED" << std::endl;
      }

      cap.release();
    } else {
      std::cout << "FAILED" << std::endl;
    }
  }

  std::cout << "=============================" << std::endl;
  std::cout << "Total cameras detected: " << count << std::endl;

  return 0;
}
