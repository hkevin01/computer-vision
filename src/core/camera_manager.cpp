#include "camera_manager.hpp"
#include <iostream>
#include <opencv2/videoio.hpp>

namespace stereo_vision {

CameraManager::CameraManager() : m_cap_left(nullptr), m_cap_right(nullptr) {}

CameraManager::~CameraManager() { closeCameras(); }

int CameraManager::detectCameras() {
  m_camera_list.clear();
  int count = 0;
  cv::VideoCapture temp_cap;
  while (temp_cap.open(count)) {
    std::string name = "Camera " + std::to_string(count);
    // A more descriptive name might be available on some platforms, but this is
    // a robust default.
    m_camera_list.push_back(name);
    temp_cap.release();
    count++;
  }
  return count;
}

bool CameraManager::openCameras(int left_cam_idx, int right_cam_idx) {
  closeCameras();
  m_cap_left = std::make_unique<cv::VideoCapture>(left_cam_idx, cv::CAP_ANY);
  m_cap_right = std::make_unique<cv::VideoCapture>(right_cam_idx, cv::CAP_ANY);

  if (!m_cap_left->isOpened() || !m_cap_right->isOpened()) {
    std::cerr << "Error: Could not open one or both cameras." << std::endl;
    closeCameras();
    return false;
  }
  return true;
}

bool CameraManager::areCamerasOpen() const {
  return m_cap_left && m_cap_left->isOpened() && m_cap_right &&
         m_cap_right->isOpened();
}

bool CameraManager::grabFrames(cv::Mat &left_frame, cv::Mat &right_frame) {
  if (!areCamerasOpen()) {
    return false;
  }
  return m_cap_left->read(left_frame) && m_cap_right->read(right_frame);
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

} // namespace stereo_vision
