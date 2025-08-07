#!/bin/bash

echo "🚀 Stereo Vision GUI Launcher"
echo "=========================="

# Function to launch GUI with different methods
launch_gui() {
    local app="$1"
    local method="$2"
    local extra_args="$3"

    echo "📋 Launching $app with $method..."

    case "$method" in
        "wayland")
            echo "🔄 Using Wayland (default)..."
            $app $extra_args &
            ;;
        "x11")
            echo "🔄 Using X11 platform..."
            QT_QPA_PLATFORM=xcb $app $extra_args &
            ;;
        "debug")
            echo "🔄 Debug mode with verbose output..."
            QT_DEBUG_PLUGINS=1 QT_QPA_PLATFORM=xcb $app $extra_args
            ;;
        *)
            echo "❌ Unknown method: $method"
            return 1
            ;;
    esac

    local pid=$!
    echo "✅ Started with PID: $pid"
    echo "💡 If no window appears, try: ./launch_gui.sh $app x11"
    echo "🔍 For debugging, try: ./launch_gui.sh $app debug"
}

# Check which applications are available
echo "🔍 Checking available applications..."

if [ -f "build/stereo_vision_app" ]; then
    echo "✅ Full GUI application: build/stereo_vision_app"
    FULL_APP="build/stereo_vision_app"
else
    echo "⚠️  Full GUI application not found (still building?)"
    FULL_APP=""
fi

if [ -f "build/stereo_vision_app_simple" ]; then
    echo "✅ Simple test application: build/stereo_vision_app_simple"
    SIMPLE_APP="build/stereo_vision_app_simple"
else
    echo "❌ Simple test application not found"
    SIMPLE_APP=""
fi

echo ""

# Determine what to launch
if [ "$1" = "full" ] && [ -n "$FULL_APP" ]; then
    launch_gui "$FULL_APP" "${2:-wayland}"
elif [ "$1" = "simple" ] && [ -n "$SIMPLE_APP" ]; then
    launch_gui "$SIMPLE_APP" "${2:-wayland}"
elif [ "$1" = "full" ] && [ -z "$FULL_APP" ]; then
    echo "❌ Full GUI application not available yet"
    if [ -n "$SIMPLE_APP" ]; then
        echo "🎯 Launching enhanced GUI with full application interface..."
        launch_gui "$SIMPLE_APP" "wayland" "--simulate-full"
    else
        echo "❌ No applications available"
        exit 1
    fi
elif [ "$1" = "debug" ]; then
    if [ -n "$FULL_APP" ]; then
        launch_gui "$FULL_APP" "debug"
    elif [ -n "$SIMPLE_APP" ]; then
        launch_gui "$SIMPLE_APP" "debug"
    else
        echo "❌ No applications available for debugging"
    fi
else
    # Auto-select best available app
    if [ -n "$FULL_APP" ]; then
        echo "🎯 Auto-launching full GUI application..."
        launch_gui "$FULL_APP" "${1:-wayland}"
    elif [ -n "$SIMPLE_APP" ]; then
        echo "🎯 Auto-launching simple test application..."
        launch_gui "$SIMPLE_APP" "${1:-wayland}"
    else
        echo "❌ No GUI applications available!"
        echo ""
        echo "📝 Usage:"
        echo "  $0                    # Auto-launch best available app"
        echo "  $0 full              # Launch full GUI application"
        echo "  $0 simple            # Launch simple test application"
        echo "  $0 x11               # Force X11 platform"
        echo "  $0 debug             # Debug mode with verbose output"
        echo ""
        echo "🛠️  Try building first: make -C build"
        exit 1
    fi
fi

echo ""
echo "🖥️  GUI Launch Summary:"
echo "   Display: $DISPLAY"
echo "   Session: $XDG_SESSION_TYPE"
echo "   Wayland: $WAYLAND_DISPLAY"
echo ""
echo "💡 Tips:"
echo "   - If no window shows, the app might have crashed or exited quickly"
echo "   - Try: ./launch_gui.sh debug to see detailed output"
echo "   - For X11 compatibility: ./launch_gui.sh x11"
echo "   - Check running processes: ps aux | grep stereo_vision"
