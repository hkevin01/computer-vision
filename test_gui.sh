#!/bin/bash

# Test script for the Stereo Vision GUI application
# This script runs the application briefly to test the new features

echo "=== Testing Stereo Vision GUI Application ==="
echo "Features to test:"
echo "1. AI Calibration menu option"
echo "2. Live Processing menu option" 
echo "3. Live Processing tab"
echo "4. Camera detection and capture"
echo ""

# Set up environment to avoid snap conflicts
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Check for cameras
echo "Available cameras:"
ls /dev/video* 2>/dev/null || echo "No cameras detected"
echo ""

# Build the application
echo "Building application..."
cd /home/kevin/Projects/computer-vision
cmake --build build --config Debug

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Starting application..."
    echo "The GUI should open with the following new features:"
    echo "- 'AI Calibration' option in the Calibration menu"
    echo "- 'Start Live Processing' option in the Tools menu"
    echo "- 'Live Processing' tab in the main interface"
    echo ""
    echo "Press Ctrl+C to close the application when testing is complete."
    echo ""
    
    # Run the application
    ./build/stereo_vision_app
else
    echo "Build failed! Check the error messages above."
    exit 1
fi
