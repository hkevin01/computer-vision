#!/bin/bash

# Default values
BUILD_DIR="build"
TARGET="all"
EXECUTABLE="stereo_vision_app"
RUN_APP=true
RUN_TESTS=false
CLEAN_BUILD=false
EXTRA_CMAKE_ARGS=""

# Function to display help message
function show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message."
    echo "  -t, --tests           Run the test suite instead of the main application."
    echo "  -c, --clean           Perform a clean build."
    echo "  --build-dir <dir>     Specify the build directory (default: build)."
    echo "  --target <target>     Specify the cmake build target (default: all)."
    echo "  --amd                 Use the AMD/HIP build configuration (build_amd)."
    echo "  --debug               Use the Debug build configuration."
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
        EXTRA_CMAKE_ARGS="-DENABLE_HIP=ON -DENABLE_CUDA=OFF"
        shift # past argument
        ;;
        --debug)
        EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug"
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

# Clean the build directory if requested
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "--- Cleaning build directory: $BUILD_DIR ---"
    rm -rf "$BUILD_DIR"
fi

# Configure the project using CMake
if [ ! -d "$BUILD_DIR" ]; then
    echo "--- Configuring project in: $BUILD_DIR ---"
    cmake -B "$BUILD_DIR" -S . ${EXTRA_CMAKE_ARGS}
fi

# Build the project
echo "--- Building project in: $BUILD_DIR (Target: $TARGET) ---"
cmake --build "$BUILD_DIR" --config Debug --target "$TARGET" --parallel $(nproc)

# Run the application or tests
if [ "$RUN_APP" = true ]; then
    echo "--- Running Application ---"
    ./"$BUILD_DIR"/"$EXECUTABLE"
elif [ "$RUN_TESTS" = true ]; then
    echo "--- Running Tests ---"
    # Use ctest to run the tests discovered by gtest_discover_tests
    cd "$BUILD_DIR" && ctest --output-on-failure
fi

echo "--- Script finished ---"