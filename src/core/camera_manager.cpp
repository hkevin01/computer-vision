#include "camera_manager.hpp"
#include <iostream>
#include <opencv2/videoio.hpp>

namespace stereo_vision {

CameraManager::CameraManager() : m_cap_left(nullptr), m_cap_right(nullptr) {}

CameraManager::~CameraManager() { closeCameras(); }

int CameraManager::detectCameras() {
  m_camera_list.clear();
  m_device_indices.clear();
  int count = 0;
  int max_test = 10; // Test up to 10 camera indices (enough for most systems)
  int consecutive_failures = 0;
  int max_consecutive_failures = 3; // Stop after 3 consecutive failures

  std::cout << "=== Camera Detection Started ===" << std::endl;
  std::cout << "Scanning for available cameras..." << std::endl;

  for (int i = 0;
       i < max_test && consecutive_failures < max_consecutive_failures; i++) {
    cv::VideoCapture temp_cap;
    std::cout << "Testing camera index " << i << "..." << std::endl;

    // Try V4L2 backend first (more reliable on Linux)
    bool opened = false;
    if (temp_cap.open(i, cv::CAP_V4L2)) {
      opened = true;
      std::cout << "  Opened with V4L2 backend" << std::endl;
    } else if (temp_cap.open(i, cv::CAP_ANY)) {
      opened = true;
      std::cout << "  Opened with ANY backend" << std::endl;
    } else {
      std::cout << "  Failed to open device" << std::endl;
    }

    if (opened) {
      // Additional check: try to capture a frame to verify it's a real capture
      // device
      cv::Mat test_frame;
      bool can_capture = temp_cap.read(test_frame);

      if (can_capture && !test_frame.empty()) {
        // Get camera properties for better identification
        double width = temp_cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = temp_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = temp_cap.get(cv::CAP_PROP_FPS);

        std::string name = "Camera " + std::to_string(count) + " (Index " +
                           std::to_string(i) + ", " +
                           std::to_string(static_cast<int>(width)) + "x" +
                           std::to_string(static_cast<int>(height)) + ")";

        m_camera_list.push_back(name);
        m_device_indices.push_back(i); // Store the actual device index
        count++;
        consecutive_failures = 0; // Reset failure counter

        std::cout << "  ✅ Working camera found at index " << i << ": " << width
                  << "x" << height << " @ " << fps << " FPS" << std::endl;
      } else {
        std::cout
            << "  ❌ Device opened but cannot capture frames (metadata device?)"
            << std::endl;
        consecutive_failures++;
      }

      temp_cap.release();
    } else {
      consecutive_failures++;
    }
  }

  std::cout << "=== Camera Detection Complete ===" << std::endl;
  std::cout << "Total usable cameras detected: " << count << std::endl;

  if (count == 0) {
    std::cout << "❌ No working cameras found!" << std::endl;
    std::cout << "Please check:" << std::endl;
    std::cout << "  - Camera is connected and powered" << std::endl;
    std::cout << "  - User is in 'video' group: groups $USER | grep video"
              << std::endl;
    std::cout << "  - Camera permissions: ls -la /dev/video*" << std::endl;
  } else if (count == 1) {
    std::cout << "⚠️  Only ONE camera detected!" << std::endl;
    std::cout << "For stereo vision, you need TWO separate physical cameras."
              << std::endl;
    std::cout << "Single camera mode will be available for testing."
              << std::endl;
  } else {
    std::cout << "✅ Multiple cameras detected - stereo vision available!"
              << std::endl;
  }

  return count;
}

bool CameraManager::openCameras(int left_cam_idx, int right_cam_idx) {
  closeCameras();

  // Convert logical indices to actual device indices
  int actual_left_idx = getDeviceIndex(left_cam_idx);
  int actual_right_idx = getDeviceIndex(right_cam_idx);

  if (actual_left_idx < 0 || actual_right_idx < 0) {
    std::cerr << "Error: Invalid camera indices provided." << std::endl;
    return false;
  }

  std::cout << "Opening cameras: logical " << left_cam_idx << " (device "
            << actual_left_idx << ") and logical " << right_cam_idx
            << " (device " << actual_right_idx << ")" << std::endl;

  // Try V4L2 backend first for better performance on Linux
  m_cap_left =
      std::make_unique<cv::VideoCapture>(actual_left_idx, cv::CAP_V4L2);
  if (!m_cap_left->isOpened()) {
    m_cap_left =
        std::make_unique<cv::VideoCapture>(actual_left_idx, cv::CAP_ANY);
  }

  m_cap_right =
      std::make_unique<cv::VideoCapture>(actual_right_idx, cv::CAP_V4L2);
  if (!m_cap_right->isOpened()) {
    m_cap_right =
        std::make_unique<cv::VideoCapture>(actual_right_idx, cv::CAP_ANY);
  }

  if (!m_cap_left->isOpened() || !m_cap_right->isOpened()) {
    std::cerr << "Error: Could not open one or both cameras." << std::endl;
    std::cerr << "Left camera (device " << actual_left_idx
              << "): " << (m_cap_left->isOpened() ? "OK" : "FAILED")
              << std::endl;
    std::cerr << "Right camera (device " << actual_right_idx
              << "): " << (m_cap_right->isOpened() ? "OK" : "FAILED")
              << std::endl;
    closeCameras();
    return false;
  }

  std::cout << "Both cameras opened successfully." << std::endl;
  return true;
}

bool CameraManager::openSingleCamera(int cam_idx) {
  closeCameras();

  // Convert logical index to actual device index
  int actual_idx = getDeviceIndex(cam_idx);

  if (actual_idx < 0) {
    std::cerr << "Error: Invalid camera index provided." << std::endl;
    return false;
  }

  std::cout << "Opening single camera: logical " << cam_idx << " (device "
            << actual_idx << ")" << std::endl;

  // Open as left camera only
  m_cap_left = std::make_unique<cv::VideoCapture>(actual_idx, cv::CAP_V4L2);
  if (!m_cap_left->isOpened()) {
    m_cap_left = std::make_unique<cv::VideoCapture>(actual_idx, cv::CAP_ANY);
  }

  if (!m_cap_left->isOpened()) {
    std::cerr << "Error: Could not open camera (device " << actual_idx << ")"
              << std::endl;
    closeCameras();
    return false;
  }

  std::cout << "Single camera opened successfully." << std::endl;
  return true;
}

bool CameraManager::areCamerasOpen() const {
  return m_cap_left && m_cap_left->isOpened() && m_cap_right &&
         m_cap_right->isOpened();
}

bool CameraManager::isAnyCameraOpen() const {
  return (m_cap_left && m_cap_left->isOpened()) ||
         (m_cap_right && m_cap_right->isOpened());
}

bool CameraManager::grabFrames(cv::Mat &left_frame, cv::Mat &right_frame) {
  if (!areCamerasOpen()) {
    return false;
  }
  return m_cap_left->read(left_frame) && m_cap_right->read(right_frame);
}

bool CameraManager::grabSingleFrame(cv::Mat &frame) {
  if (m_cap_left && m_cap_left->isOpened()) {
    return m_cap_left->read(frame);
  } else if (m_cap_right && m_cap_right->isOpened()) {
    return m_cap_right->read(frame);
  }
  return false;
}

void CameraManager::closeCameras() {
  if (m_cap_left) {
    m_cap_left->release();
    m_cap_left.reset();
  }
  if (m_cap_right) {
    m_cap_right->release();
    m_cap_right.reset();
  }
}

const std::vector<std::string> &CameraManager::getCameraList() const {
  return m_camera_list;
}

int CameraManager::getDeviceIndex(int logical_index) const {
  if (logical_index >= 0 &&
      logical_index < static_cast<int>(m_device_indices.size())) {
    return m_device_indices[logical_index];
  }
  return -1; // Invalid index
}

} // namespace stereo_vision
