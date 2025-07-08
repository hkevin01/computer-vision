#!/bin/bash

# Default values
BUILD_DIR="build"
TARGET="all"
EXECUTABLE="stereo_vision_app"
RUN_APP=true
RUN_TESTS=false
CLEAN_BUILD=false
FORCE_RECONFIG=false
EXTRA_CMAKE_ARGS=""

# Function to display help message
function show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Description:"
    echo "  Build and run the Stereo Vision 3D Point Cloud application."
    echo "  This script automatically detects GPU backends and configures the build accordingly."
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message."
    echo "  -t, --tests           Run the test suite instead of the main application."
    echo "  -c, --clean           Perform a clean build."
    echo "  -r, --force-reconfig  Force CMake reconfiguration even if cache exists."
    echo "  --build-dir <dir>     Specify the build directory (default: build)."
    echo "  --target <target>     Specify the cmake build target (default: all)."
    echo "  --amd                 Use the AMD/HIP build configuration (build_amd)."
    echo "  --debug               Use the Debug build configuration."
    echo "  --cpu-only            Disable GPU backends (CUDA and HIP)."
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and run main application"
    echo "  $0 --tests            # Run test suite"
    echo "  $0 --amd --clean      # Clean AMD/HIP build"
    echo "  $0 --force-reconfig   # Fix build issues"
    echo ""
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
        show_help
        ;;
        -t|--tests)
        RUN_TESTS=true
        RUN_APP=false
        EXECUTABLE="run_tests"
        shift # past argument
        ;;
        -c|--clean)
        CLEAN_BUILD=true
        shift # past argument
        ;;
        -r|--force-reconfig)
        FORCE_RECONFIG=true
        shift # past argument
        ;;
        --build-dir)
        BUILD_DIR="$2"
        shift # past argument
        shift # past value
        ;;
        --target)
        TARGET="$2"
        shift # past argument
        shift # past value
        ;;
        --amd)
        BUILD_DIR="build_amd"
        EXTRA_CMAKE_ARGS="-DUSE_HIP=ON -DUSE_CUDA=OFF"
        shift # past argument
        ;;
        --debug)
        EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug"
        shift # past argument
        ;;
        --cpu-only)
        EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DUSE_HIP=OFF -DUSE_CUDA=OFF"
        shift # past argument
        ;;
        *)
        echo "Unknown option: $1"
        show_help
        ;;
    esac
done

# --- Script Execution ---

set -e # Exit immediately if a command exits with a non-zero status.

# Ensure we're in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verify we're in the correct project directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "ERROR: CMakeLists.txt not found. Please run this script from the project root directory."
    echo "Current directory: $(pwd)"
    exit 1
fi

# Clean the build directory if requested
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "--- Cleaning build directory: $BUILD_DIR ---"
    rm -rf "$BUILD_DIR"
fi

# Configure the project using CMake
if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ ! -f "$BUILD_DIR/Makefile" ] || [ "$FORCE_RECONFIG" = true ]; then
    echo "--- Configuring project in: $BUILD_DIR ---"
    # Remove incomplete build directory if it exists
    if [ -d "$BUILD_DIR" ] && ([ ! -f "$BUILD_DIR/Makefile" ] || [ "$FORCE_RECONFIG" = true ]); then
        echo "--- Removing incomplete/old build directory ---"
        rm -rf "$BUILD_DIR"
    fi
    cmake -B "$BUILD_DIR" -S . ${EXTRA_CMAKE_ARGS}
    
    # Verify that configuration completed successfully
    if [ ! -f "$BUILD_DIR/Makefile" ] && [ ! -f "$BUILD_DIR/build.ninja" ]; then
        echo "ERROR: CMake configuration failed - no build system generated"
        exit 1
    fi
fi

# Build the project
echo "--- Building project in: $BUILD_DIR (Target: $TARGET) ---"
if [ "$RUN_TESTS" = true ]; then
    echo "Building test dependencies first..."
    # Build core components first for tests
    if ! timeout 300 cmake --build "$BUILD_DIR" --config Debug --target stereo_vision_core --parallel $(nproc); then
        echo "ERROR: Core build failed or timed out after 5 minutes."
        exit 1
    fi
    echo "Core build completed, building tests..."
fi

if ! timeout 600 cmake --build "$BUILD_DIR" --config Debug --target "$TARGET" --parallel $(nproc); then
    echo "ERROR: Build failed or timed out. Try using --force-reconfig to fix configuration issues."
    exit 1
fi

echo "--- Build completed successfully ---"

# Run the application or tests
if [ "$RUN_APP" = true ]; then
    echo "--- Running Application ---"
    if [ -f "$BUILD_DIR/$EXECUTABLE" ]; then
        ./"$BUILD_DIR"/"$EXECUTABLE"
    else
        echo "ERROR: Application executable not found: $BUILD_DIR/$EXECUTABLE"
        echo "Available executables in build directory:"
        find "$BUILD_DIR" -type f -executable -name "*stereo*" 2>/dev/null || echo "  No stereo-related executables found"
        exit 1
    fi
elif [ "$RUN_TESTS" = true ]; then
    echo "--- Running Tests ---"
    if [ -d "$BUILD_DIR" ]; then
        cd "$BUILD_DIR" && ctest --output-on-failure
    else
        echo "ERROR: Build directory not found: $BUILD_DIR"
        exit 1
    fi
fi

echo "--- Script finished ---"