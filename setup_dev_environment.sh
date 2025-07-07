#!/bin/bash

# Stereo Vision Project Setup Script
# This script configures the optimal development environment for the C++ computer vision project

set -e  # Exit on any error

echo "========================================"
echo "Stereo Vision Project Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for safety reasons"
   exit 1
fi

# Detect system
print_status "Detecting system configuration..."
OS=$(lsb_release -si 2>/dev/null || echo "Unknown")
ARCH=$(uname -m)
print_status "OS: $OS, Architecture: $ARCH"

# Check for required compilers
print_status "Checking for C++ compilers..."

# Check GCC
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "GCC found: version $GCC_VERSION"
    HAS_GCC=true
else
    print_warning "GCC not found"
    HAS_GCC=false
fi

# Check Clang
if command -v clang &> /dev/null; then
    CLANG_VERSION=$(clang --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "Clang found: version $CLANG_VERSION"
    HAS_CLANG=true
else
    print_warning "Clang not found"
    HAS_CLANG=false
fi

# Check CUDA
print_status "Checking for CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | grep -oE '[0-9]+\.[0-9]+')
    print_success "CUDA found: version $CUDA_VERSION"
    HAS_CUDA=true
else
    print_warning "CUDA not found"
    HAS_CUDA=false
fi

# Check CMake
print_status "Checking for CMake..."
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    print_success "CMake found: version $CMAKE_VERSION"
    HAS_CMAKE=true
else
    print_warning "CMake not found"
    HAS_CMAKE=false
fi

# Function to install packages
install_packages() {
    print_status "Installing required packages..."
    
    if [[ "$OS" == "Ubuntu" ]] || [[ "$OS" == "Debian" ]]; then
        sudo apt update
        
        # Essential build tools
        sudo apt install -y build-essential cmake git pkg-config
        
        # OpenCV dependencies
        sudo apt install -y libopencv-dev libopencv-contrib-dev
        
        # PCL dependencies
        sudo apt install -y libpcl-dev
        
        # GTK3 for GUI
        sudo apt install -y libgtk-3-dev
        
        # Additional useful packages
        sudo apt install -y \
            libeigen3-dev \
            libflann-dev \
            libvtk9-dev \
            qtbase5-dev \
            libqt5opengl5-dev
            
    elif [[ "$OS" == "CentOS" ]] || [[ "$OS" == "RHEL" ]] || [[ "$OS" == "Fedora" ]]; then
        if command -v dnf &> /dev/null; then
            PKG_MGR="dnf"
        else
            PKG_MGR="yum"
        fi
        
        sudo $PKG_MGR groupinstall -y "Development Tools"
        sudo $PKG_MGR install -y cmake git pkg-config
        sudo $PKG_MGR install -y opencv-devel pcl-devel gtk3-devel
    else
        print_error "Unsupported Linux distribution: $OS"
        print_status "Please install dependencies manually:"
        print_status "- OpenCV 4.5+"
        print_status "- PCL 1.12+"
        print_status "- GTK3"
        print_status "- CMake 3.18+"
    fi
}

# Check if user wants to install packages
if ! $HAS_CMAKE || ! command -v pkg-config &> /dev/null; then
    echo
    read -p "Install required packages? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_packages
    else
        print_warning "Skipping package installation. You may need to install dependencies manually."
    fi
fi

# Choose optimal compiler
print_status "Selecting optimal compiler configuration..."

CHOSEN_COMPILER=""
CHOSEN_COMPILER_PATH=""

if $HAS_CLANG && [[ "$CLANG_VERSION" > "10.0" ]]; then
    CHOSEN_COMPILER="Clang"
    CHOSEN_COMPILER_PATH=$(which clang++)
    print_success "Selected Clang $CLANG_VERSION (recommended for this project)"
elif $HAS_GCC && [[ "$GCC_VERSION" > "9.0" ]]; then
    CHOSEN_COMPILER="GCC"
    CHOSEN_COMPILER_PATH=$(which g++)
    print_success "Selected GCC $GCC_VERSION"
else
    print_error "No suitable C++ compiler found (need GCC 9+ or Clang 10+)"
    exit 1
fi

# Create VS Code configuration
print_status "Creating VS Code configuration..."

mkdir -p .vscode

# Create c_cpp_properties.json
cat > .vscode/c_cpp_properties.json << EOF
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "\${workspaceFolder}/**",
                "\${workspaceFolder}/include",
                "/usr/include/opencv4",
                "/usr/include/pcl-1.12",
                "/usr/include/eigen3",
                "/usr/include/gtk-3.0",
                "/usr/include/glib-2.0",
                "/usr/lib/x86_64-linux-gnu/glib-2.0/include",
                "/usr/local/cuda/include"
            ],
            "defines": [],
            "compilerPath": "$CHOSEN_COMPILER_PATH",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
EOF

# Create settings.json
cat > .vscode/settings.json << EOF
{
    "cmake.configureOnOpen": true,
    "cmake.buildDirectory": "\${workspaceFolder}/build",
    "cmake.generator": "Unix Makefiles",
    "cmake.preferredGenerators": [
        "Unix Makefiles",
        "Ninja"
    ],
    "cmake.configureArgs": [
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ],
    "files.associations": {
        "*.cu": "cuda-cpp",
        "*.cuh": "cuda-cpp"
    },
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "C_Cpp.default.compilerPath": "$CHOSEN_COMPILER_PATH"
}
EOF

# Create launch.json for debugging
cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Stereo Vision App",
            "type": "cppdbg",
            "request": "launch",
            "program": "\${workspaceFolder}/build/stereo_vision_app",
            "args": ["--help"],
            "stopAtEntry": false,
            "cwd": "\${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "cmake: build",
            "miDebuggerPath": "/usr/bin/gdb",
            "logging": {
                "moduleLoad": false,
                "programOutput": true,
                "engineLogging": false
            }
        }
    ]
}
EOF

# Create tasks.json
cat > .vscode/tasks.json << EOF
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cmake: configure",
            "type": "shell",
            "command": "cmake",
            "args": [
                "-B",
                "build",
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            },
            "problemMatcher": []
        },
        {
            "label": "cmake: build",
            "type": "shell",
            "command": "cmake",
            "args": [
                "--build",
                "build",
                "--config",
                "Debug",
                "--parallel",
                "\$(nproc)"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            },
            "problemMatcher": [
                "\$gcc"
            ],
            "dependsOn": "cmake: configure"
        },
        {
            "label": "cmake: clean",
            "type": "shell",
            "command": "cmake",
            "args": [
                "--build",
                "build",
                "--target",
                "clean"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            }
        }
    ]
}
EOF

# Create extensions.json with recommended extensions
cat > .vscode/extensions.json << EOF
{
    "recommendations": [
        "ms-vscode.cpptools",
        "ms-vscode.cmake-tools",
        "twxs.cmake",
        "ms-vscode.cpptools-extension-pack",
        "llvm-vs-code-extensions.vscode-clangd",
        "nvidia.nsight-vscode-edition"
    ]
}
EOF

print_success "VS Code configuration created"

# Create build script
print_status "Creating build scripts..."

cat > build.sh << EOF
#!/bin/bash
set -e

echo "Building Stereo Vision Project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
make -j\$(nproc)

echo "Build completed successfully!"
echo "Executable: ./stereo_vision_app"
echo "Test executable: ./test_stereo_vision"
EOF

chmod +x build.sh

cat > build_debug.sh << EOF
#!/bin/bash
set -e

echo "Building Stereo Vision Project (Debug)..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
make -j\$(nproc)

echo "Debug build completed successfully!"
echo "Executable: ./stereo_vision_app"
echo "Test executable: ./test_stereo_vision"
EOF

chmod +x build_debug.sh

# Create clean script
cat > clean.sh << EOF
#!/bin/bash
echo "Cleaning build artifacts..."
rm -rf build/
echo "Clean completed!"
EOF

chmod +x clean.sh

print_success "Build scripts created"

# Download sample data
print_status "Setting up sample data..."

mkdir -p data/sample_images/left
mkdir -p data/sample_images/right

# Create a simple script to download sample stereo images
cat > scripts/download_sample_data.sh << EOF
#!/bin/bash

echo "Downloading sample stereo images..."

# Create sample data directories
mkdir -p data/sample_images/left
mkdir -p data/sample_images/right
mkdir -p data/calibration

# Note: Add URLs to real stereo datasets here
# For now, create placeholder files
echo "Sample stereo image datasets:"
echo "1. Middlebury Stereo Dataset: https://vision.middlebury.edu/stereo/data/"
echo "2. KITTI Dataset: http://www.cvlibs.net/datasets/kitti/"
echo "3. ETH3D Dataset: https://www.eth3d.net/"

echo "Please download sample stereo images and place them in:"
echo "  data/sample_images/left/"
echo "  data/sample_images/right/"
echo "  data/calibration/"

# Create sample calibration checkerboard pattern
echo "Creating sample checkerboard pattern..."
python3 -c "
import cv2
import numpy as np

# Create 9x6 checkerboard pattern
board_w, board_h = 9, 6
square_size = 100  # pixels

img_w = board_w * square_size
img_h = board_h * square_size

# Create checkerboard
img = np.zeros((img_h, img_w), dtype=np.uint8)
for i in range(board_h):
    for j in range(board_w):
        if (i + j) % 2 == 0:
            y1, y2 = i * square_size, (i + 1) * square_size
            x1, x2 = j * square_size, (j + 1) * square_size
            img[y1:y2, x1:x2] = 255

cv2.imwrite('data/calibration/checkerboard_9x6.png', img)
print('Checkerboard pattern saved to data/calibration/checkerboard_9x6.png')
" 2>/dev/null || echo "Python3/OpenCV not available for checkerboard generation"

echo "Sample data setup completed!"
EOF

chmod +x scripts/download_sample_data.sh

# Create development environment info
cat > DEV_ENVIRONMENT.md << EOF
# Development Environment Setup

## Compiler Configuration
- **Selected Compiler**: $CHOSEN_COMPILER
- **Compiler Path**: $CHOSEN_COMPILER_PATH
- **C++ Standard**: C++17
- **CUDA Available**: $HAS_CUDA

## Build System
- **CMake Version**: $CMAKE_VERSION
- **Build Directory**: build/
- **Default Build Type**: Debug (for development)

## VS Code Configuration
The following VS Code files have been configured:
- **.vscode/c_cpp_properties.json**: IntelliSense configuration
- **.vscode/settings.json**: Workspace settings
- **.vscode/launch.json**: Debug configuration
- **.vscode/tasks.json**: Build tasks
- **.vscode/extensions.json**: Recommended extensions

## Build Scripts
- **build.sh**: Release build
- **build_debug.sh**: Debug build
- **clean.sh**: Clean build artifacts

## Quick Start
1. Install recommended VS Code extensions
2. Run: \`./build.sh\` to build the project
3. Run: \`./build/stereo_vision_app --help\` to see usage options

## Dependencies Status
$(if $HAS_CUDA; then echo "✅ CUDA: Available"; else echo "❌ CUDA: Not available"; fi)
$(if command -v pkg-config --modversion opencv4 &>/dev/null; then echo "✅ OpenCV: $(pkg-config --modversion opencv4)"; else echo "❓ OpenCV: Check installation"; fi)
$(if command -v pkg-config --modversion pcl_common &>/dev/null; then echo "✅ PCL: Available"; else echo "❓ PCL: Check installation"; fi)
$(if command -v pkg-config --modversion gtk+-3.0 &>/dev/null; then echo "✅ GTK3: $(pkg-config --modversion gtk+-3.0)"; else echo "❓ GTK3: Check installation"; fi)
EOF

print_success "Development environment documentation created"

# Final summary
echo
echo "========================================"
print_success "Setup completed successfully!"
echo "========================================"
echo
print_status "What was configured:"
echo "  ✅ VS Code configuration (.vscode/)"
echo "  ✅ Build scripts (build.sh, build_debug.sh, clean.sh)"
echo "  ✅ Sample data structure (data/)"
echo "  ✅ Development environment documentation"
echo
print_status "Next steps:"
echo "  1. Open the project in VS Code"
echo "  2. Install recommended extensions when prompted"
echo "  3. Run './build.sh' to build the project"
echo "  4. Run './scripts/download_sample_data.sh' to get sample data"
echo
print_status "To start development:"
echo "  code .  # Open in VS Code"
echo "  ./build.sh  # Build the project"
echo
if ! $HAS_CUDA; then
    print_warning "CUDA not detected. GPU acceleration will not be available."
    print_status "To install CUDA, visit: https://developer.nvidia.com/cuda-downloads"
fi

echo "Happy coding! 🚀"
