# Test Programs

This directory contains standalone test programs for various components of the stereo vision system.

## 📁 Test Programs Overview

### Camera Testing
- **test_camera_manager** - Test the camera manager functionality
- **test_camera_manager.cpp** - Source code for camera manager tests
- **test_camera_manager_simple** - Simplified camera manager test
- **test_camera_manager_simple.cpp** - Simplified test source code
- **test_cameras** - Basic camera detection test
- **test_cameras.cpp** - Camera detection source code
- **test_direct_camera** - Direct camera access test
- **test_direct_camera.cpp** - Direct camera test source code
- **test_single_camera** - Single camera functionality test
- **test_single_camera.cpp** - Single camera test source code

### GUI Testing
- **test_qt_simple** - Simple Qt GUI test
- **test_qt.cpp** - Qt testing source code

## 🧪 Running Tests

### Camera Tests
```bash
# From project root directory
cd test_programs/

# Test camera detection
./test_cameras

# Test camera manager
./test_camera_manager_simple

# Test direct camera access
./test_direct_camera

# Test single camera mode
./test_single_camera
```

### Qt GUI Tests
```bash
# Test Qt functionality
./test_qt_simple
```

## 🔨 Compiling Tests

If you need to recompile any test programs:

```bash
# Example: Recompile camera manager test
cd /home/kevin/Projects/computer-vision
g++ -std=c++17 -I include -o test_programs/test_camera_manager_simple \
    test_programs/test_camera_manager_simple.cpp \
    src/core/camera_manager.cpp \
    $(pkg-config --cflags --libs opencv4)
```

## 📊 Test Results

### Expected Outputs

#### Camera Detection Tests
- Should detect available cameras (currently 1 NexiGo camera)
- Display camera capabilities and resolution
- Show appropriate warnings for stereo mode requirements

#### GUI Tests
- Verify Qt functionality and widget creation
- Test basic GUI components

## 🔍 Troubleshooting

### Common Issues
1. **Permission errors**: Ensure user is in 'video' group
2. **Library conflicts**: May encounter snap-related library issues
3. **Camera not found**: Check camera connections and permissions

### Debug Information
The test programs provide detailed output for debugging:
- Camera enumeration details
- OpenCV backend information
- Error messages with specific failure reasons

## 📝 Notes

- These are standalone test programs separate from the main application
- Use these for debugging specific component issues
- The main application (`stereo_vision_app`) includes more comprehensive testing
- Test programs help isolate issues to specific components
