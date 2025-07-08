#pragma once

#include <memory>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

namespace stereo_vision {

class CameraManager {
public:
  CameraManager();
  ~CameraManager();

  // Detects available cameras and returns the number of cameras found.
  int detectCameras();

  // Opens the specified cameras.
  bool openCameras(int left_cam_idx, int right_cam_idx);

  // Checks if both cameras are opened successfully.
  bool areCamerasOpen() const;

  // Grabs a frame from each camera.
  bool grabFrames(cv::Mat &left_frame, cv::Mat &right_frame);

  // Releases the cameras.
  void closeCameras();

  // Get the list of available camera names/descriptions (platform-dependent).
  const std::vector<std::string> &getCameraList() const;

private:
  std::unique_ptr<cv::VideoCapture> m_cap_left;
  std::unique_ptr<cv::VideoCapture> m_cap_right;
  std::vector<std::string> m_camera_list;
};

} // namespace stereo_vision
