# Webcam Capture Integration

## Overview

The Stereo Vision 3D Point Cloud Generator now includes comprehensive webcam capture functionality, allowing users to directly capture stereo image pairs from connected USB or built-in cameras.

## Features

### Camera Management
- **Automatic Device Detection**: Detects all available camera devices on the system
- **Device Selection**: Independent selection of left and right camera devices
- **Connection Testing**: Verify camera connections before starting capture
- **Live Preview**: Real-time preview from both cameras simultaneously

### Capture Modes
- **Individual Capture**: Capture left or right images independently
- **Synchronized Capture**: Capture perfectly timed stereo pairs
- **Live Streaming**: Continuous preview at ~30 FPS
- **Flexible Formats**: Save as PNG, JPEG, BMP, or TIFF

## User Interface

### Menu Integration
The webcam capture functionality is integrated into the main menu under `File`:

- `Select Cameras...` (Ctrl+Shift+C) - Open camera selection dialog
- `Start Webcam Capture` (Ctrl+Shift+S) - Begin live capture
- `Stop Webcam Capture` (Ctrl+Shift+T) - End capture session
- `Capture Left Image` (L) - Save current left frame
- `Capture Right Image` (R) - Save current right frame
- `Capture Stereo Pair` (Space) - Save synchronized pair

### Camera Selector Dialog
- **Device Lists**: Dropdown menus for left and right camera selection
- **Status Indicators**: Real-time connection status for each camera
- **Test Function**: Verify camera connectivity before proceeding
- **Preview Toggle**: Enable/disable live preview during selection
- **Refresh Button**: Re-scan for newly connected devices

## Usage Workflow

### 1. Camera Setup
1. Connect your stereo camera setup (two USB cameras or built-in + USB)
2. Launch the application
3. Navigate to `File → Select Cameras...`
4. Choose appropriate devices for left and right cameras
5. Click "Test Cameras" to verify connections
6. Enable preview to see live feed
7. Click "OK" to confirm selection

### 2. Live Capture
1. Use `File → Start Webcam Capture` or Ctrl+Shift+S
2. Live preview appears in the Left/Right image tabs
3. Both cameras stream simultaneously
4. Status bar shows capture status and frame rate

### 3. Image Capture
- **Quick Capture**: Press L/R keys for individual images
- **Stereo Capture**: Press Space for synchronized pairs
- **Save Dialog**: Choose location and format for saved images
- **Auto-naming**: Files automatically timestamped (YYYYMMDD_HHMMSS)

### 4. Stop Capture
1. Use `File → Stop Webcam Capture` or Ctrl+Shift+T
2. Cameras are released and preview stops
3. All captured images remain loaded for processing

## Technical Implementation

### Architecture
- **CameraManager**: Core camera interface and device management
- **CameraSelectorDialog**: GUI for device selection and testing
- **MainWindow Integration**: Menu actions and live preview integration
- **OpenCV Backend**: Uses cv::VideoCapture for cross-platform compatibility

### Camera Management Flow
```cpp
// Device detection
int numCameras = cameraManager->detectCameras();

// Open selected cameras
bool success = cameraManager->openCameras(leftIndex, rightIndex);

// Capture synchronized frames
cv::Mat leftFrame, rightFrame;
bool captured = cameraManager->grabFrames(leftFrame, rightFrame);

// Release cameras
cameraManager->closeCameras();
```

### Synchronization
- Frame-level synchronization ensures stereo pairs are captured simultaneously
- Minimal latency between left and right captures
- Buffer management prevents frame dropping
- Timestamp correlation for precise synchronization

## Configuration Options

### Frame Rate
- Preview: ~10 FPS (adjustable via timer interval)
- Capture: Hardware maximum (typically 30 FPS)
- Configurable through dialog settings

### Resolution
- Automatic detection of optimal camera resolution
- Support for common formats: 640x480, 800x600, 1024x768, 1280x720
- Manual override available through camera properties

### File Format
- **PNG**: Lossless compression, best quality
- **JPEG**: Lossy compression, smaller files
- **BMP**: Uncompressed, fastest save
- **TIFF**: Lossless, professional quality

## Troubleshooting

### Common Issues

#### No Cameras Detected
- Verify USB connections are secure
- Check camera permissions (Linux: udev rules)
- Restart application if cameras were connected after launch
- Use "Refresh" button to re-scan devices

#### Camera Connection Failed
- Another application may be using the camera
- Check USB bandwidth (use USB 3.0 if available)
- Try different USB ports
- Restart the camera devices

#### Poor Frame Rate
- Close other applications using cameras
- Reduce preview resolution
- Use USB 3.0 ports for better bandwidth
- Check system performance and CPU usage

#### Synchronization Issues
- Ensure both cameras have similar specifications
- Use identical camera models when possible
- Check for hardware timing differences
- Verify USB bus bandwidth availability

### Hardware Recommendations

#### Camera Setup
- **Stereo Rig**: Fixed baseline distance (typically 6-12cm)
- **Identical Cameras**: Same model and specifications preferred
- **USB 3.0**: For higher frame rates and resolution
- **Good Lighting**: Adequate illumination for both cameras

#### Computer Requirements
- **USB Bandwidth**: Sufficient for dual camera streams
- **Processing Power**: Real-time preview requires adequate CPU
- **Memory**: Sufficient RAM for frame buffering
- **Graphics**: GPU acceleration improves performance

## Integration with Stereo Processing

### Workflow Integration
1. **Capture**: Use webcam capture to obtain stereo pairs
2. **Calibration**: Use captured images for camera calibration
3. **Processing**: Generate disparity maps from captured pairs
4. **Visualization**: View resulting 3D point clouds

### Calibration Workflow
1. Capture multiple stereo pairs of checkerboard patterns
2. Use `Process → Calibrate Cameras...` with captured images
3. Save calibration parameters for future processing
4. Apply calibration to new captures automatically

### Real-time Processing
- Live captured images automatically loaded into processing pipeline
- Immediate disparity map generation after stereo capture
- Real-time point cloud updates with noise filtering
- Interactive 3D visualization of captured scenes

## Future Enhancements

### Planned Features
- **Multiple Camera Support**: Support for more than two cameras
- **Advanced Synchronization**: Hardware synchronization support
- **Video Recording**: Continuous stereo video capture
- **Auto-calibration**: Real-time calibration during capture
- **Network Cameras**: Support for IP cameras and network streams

### Performance Improvements
- **Hardware Acceleration**: GPU-accelerated frame processing
- **Parallel Capture**: Multi-threaded camera handling
- **Buffer Optimization**: Improved memory management
- **Format Optimization**: Hardware-accelerated encoding

---

*For technical support or feature requests related to webcam capture, please refer to the main documentation or open an issue in the project repository.*
