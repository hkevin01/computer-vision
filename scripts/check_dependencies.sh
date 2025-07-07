#!/bin/bash

# Dependency checker script for the Stereo Vision project
# This script checks and reports the status of all required dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}$1${NC}"
    echo "=================================="
}

check_command() {
    local cmd=$1
    local name=$2
    local required=$3
    
    if command -v "$cmd" &> /dev/null; then
        local version=""
        case $cmd in
            "gcc"|"g++")
                version=$(gcc --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
                ;;
            "clang"|"clang++")
                version=$(clang --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
                ;;
            "cmake")
                version=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
                ;;
            "nvcc")
                version=$(nvcc --version | grep "release" | grep -oE '[0-9]+\.[0-9]+')
                ;;
        esac
        
        if [ -n "$version" ]; then
            echo -e "✅ ${name}: ${GREEN}Found${NC} (version $version)"
        else
            echo -e "✅ ${name}: ${GREEN}Found${NC}"
        fi
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "❌ ${name}: ${RED}Not found (REQUIRED)${NC}"
        else
            echo -e "⚠️  ${name}: ${YELLOW}Not found (optional)${NC}"
        fi
        return 1
    fi
}

check_library() {
    local pkg=$1
    local name=$2
    local required=$3
    
    if pkg-config --exists "$pkg" 2>/dev/null; then
        local version=$(pkg-config --modversion "$pkg" 2>/dev/null)
        echo -e "✅ ${name}: ${GREEN}Found${NC} (version $version)"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "❌ ${name}: ${RED}Not found (REQUIRED)${NC}"
        else
            echo -e "⚠️  ${name}: ${YELLOW}Not found (optional)${NC}"
        fi
        return 1
    fi
}

print_header "System Information"
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2- || uname -s)"
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo

print_header "Build Tools"
check_command "gcc" "GCC Compiler" "true"
check_command "g++" "G++ Compiler" "true"
check_command "clang" "Clang Compiler" "false"
check_command "clang++" "Clang++ Compiler" "false"
check_command "cmake" "CMake" "true"
check_command "make" "Make" "true"
check_command "ninja" "Ninja" "false"
echo

print_header "CUDA Environment"
check_command "nvcc" "NVIDIA CUDA Compiler" "false"
check_command "nvidia-smi" "NVIDIA Driver" "false"

if command -v nvidia-smi &> /dev/null; then
    echo
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  GPU: $line"
    done
fi
echo

print_header "Computer Vision Libraries"
check_library "opencv4" "OpenCV" "true"
check_library "pcl_common" "Point Cloud Library (PCL)" "true"
check_library "eigen3" "Eigen3" "false"
echo

print_header "GUI Libraries"
check_library "gtk+-3.0" "GTK3" "true"
check_library "Qt5Core" "Qt5" "false"
echo

print_header "Additional Tools"
check_command "git" "Git" "true"
check_command "pkg-config" "pkg-config" "true"
check_command "gdb" "GNU Debugger" "false"
check_command "valgrind" "Valgrind" "false"
echo

print_header "Python Environment (for utilities)"
check_command "python3" "Python 3" "false"
if command -v python3 &> /dev/null; then
    python3 -c "import cv2; print(f'  OpenCV Python: {cv2.__version__}')" 2>/dev/null || echo "  OpenCV Python: Not available"
    python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: Not available"
fi
echo

print_header "Development Environment Status"

# Check if this is a proper development setup
dev_score=0
total_checks=8

if command -v cmake &> /dev/null; then ((dev_score++)); fi
if command -v gcc &> /dev/null || command -v clang &> /dev/null; then ((dev_score++)); fi
if pkg-config --exists opencv4 2>/dev/null; then ((dev_score++)); fi
if pkg-config --exists pcl_common 2>/dev/null; then ((dev_score++)); fi
if pkg-config --exists gtk+-3.0 2>/dev/null; then ((dev_score++)); fi
if command -v git &> /dev/null; then ((dev_score++)); fi
if command -v pkg-config &> /dev/null; then ((dev_score++)); fi
if command -v nvcc &> /dev/null; then ((dev_score++)); fi

percentage=$((dev_score * 100 / total_checks))

if [ $percentage -ge 80 ]; then
    echo -e "🎉 Development environment: ${GREEN}Excellent${NC} ($percentage% complete)"
    echo "   Ready for development!"
elif [ $percentage -ge 60 ]; then
    echo -e "👍 Development environment: ${YELLOW}Good${NC} ($percentage% complete)"
    echo "   Most dependencies available, minor issues to resolve"
elif [ $percentage -ge 40 ]; then
    echo -e "⚠️  Development environment: ${YELLOW}Partial${NC} ($percentage% complete)"
    echo "   Several dependencies missing, installation recommended"
else
    echo -e "❌ Development environment: ${RED}Incomplete${NC} ($percentage% complete)"
    echo "   Major dependencies missing, run setup script"
fi

echo
echo "💡 To fix missing dependencies, run: ./setup_dev_environment.sh"
echo "📚 For manual installation, see: README.md"
