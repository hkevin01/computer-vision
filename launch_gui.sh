#!/bin/bash

# Launcher script to bypass snap conflicts
echo "=== Stereo Vision GUI Launcher ==="
echo "Attempting to launch GUI with clean environment..."

# Stop snap services temporarily
echo "Temporarily disabling snap services..."
sudo systemctl stop snapd.socket snapd.service 2>/dev/null

# Set clean environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0

# Launch the application
echo "Launching stereo_vision_app..."
echo "If successful, the GUI should appear in a new window."
echo "Press Ctrl+C to close if needed."
echo ""

./build/stereo_vision_app

# Restart snap services
echo ""
echo "Restarting snap services..."
sudo systemctl start snapd.socket snapd.service 2>/dev/null
echo "Done."
