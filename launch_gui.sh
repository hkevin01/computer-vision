#!/bin/bash

# Launcher script to bypass snap conflicts
echo "=== Stereo Vision GUI Launcher ==="
echo "Launching GUI with isolated environment to avoid snap conflicts..."

# Create a completely clean environment, excluding all snap paths and variables
CLEAN_PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/rocm/bin"
CLEAN_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib:/lib:/opt/rocm/lib:/opt/rocm/lib64"
CLEAN_XDG_DATA_DIRS="/usr/local/share:/usr/share"
CLEAN_XDG_CONFIG_DIRS="/etc/xdg"

# Launch the application
echo "Launching stereo_vision_app..."
echo "Using isolated environment (no snap contamination)..."
echo "If successful, the GUI should appear in a new window."
echo "Press Ctrl+C to close if needed."
echo ""

# Launch with completely clean environment
env -i \
    PATH="$CLEAN_PATH" \
    LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" \
    XDG_DATA_DIRS="$CLEAN_XDG_DATA_DIRS" \
    XDG_CONFIG_DIRS="$CLEAN_XDG_CONFIG_DIRS" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
    QT_QPA_PLATFORM=xcb \
    DISPLAY="${DISPLAY:-:0}" \
    XAUTHORITY="${XAUTHORITY}" \
    HOME="$HOME" \
    USER="$USER" \
    TERM="$TERM" \
    PWD="$(pwd)" \
    ./build/stereo_vision_app

echo ""
echo "GUI session ended."
