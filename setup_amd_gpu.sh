#!/bin/bash

# AMD GPU Setup Script for Computer Vision Project
# This script adds ROCm/HIP support for AMD GPUs

set -e

echo "=========================================="
echo "AMD GPU (ROCm/HIP) Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Detect AMD GPU
detect_amd_gpu() {
    print_header "Detecting AMD GPU..."
    
    if lspci | grep -i amd > /dev/null 2>&1 || lspci | grep -i ati > /dev/null 2>&1; then
        print_status "AMD GPU detected:"
        lspci | grep -i -E "(amd|ati)"
        return 0
    else
        print_error "No AMD GPU detected!"
        exit 1
    fi
}

# Install ROCm
install_rocm() {
    print_header "Installing AMD ROCm..."
    
    if command -v hipcc > /dev/null 2>&1; then
        print_status "ROCm already installed:"
        hipcc --version
        return 0
    fi
    
    print_status "Adding AMD ROCm repository..."
    
    # Add ROCm repository key
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    
    # Add repository
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
    
    sudo apt update
    
    # Install ROCm development packages
    sudo apt install -y \
        rocm-dev \
        hip-dev \
        hipblas-dev \
        rocrand-dev \
        rocfft-dev \
        rocsparse-dev \
        rocsolver-dev \
        rocthrust-dev
    
    # Add current user to render and video groups
    sudo usermod -a -G render,video $USER
    
    print_status "ROCm installed successfully!"
}

# Configure environment
configure_environment() {
    print_header "Configuring ROCm environment..."
    
    # Add ROCm paths to bashrc if not already present
    if ! grep -q "/opt/rocm/bin" ~/.bashrc; then
        echo '' >> ~/.bashrc
        echo '# ROCm environment variables' >> ~/.bashrc
        echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export HIP_PATH=/opt/rocm' >> ~/.bashrc
        print_status "Added ROCm paths to ~/.bashrc"
    fi
    
    # Create ROCm-specific build script
    cat > build_amd.sh << 'EOF'
#!/bin/bash
# Build script optimized for AMD GPUs using HIP

set -e

mkdir -p build_amd
cd build_amd

echo "Configuring for AMD GPU with HIP support..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_HIP=ON \
    -DUSE_CUDA=OFF \
    -DCMAKE_PREFIX_PATH="/opt/rocm" \
    -DHIP_ROOT_DIR="/opt/rocm" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Building with HIP support..."
make -j$(nproc)

echo "Build completed! Executable: ./stereo_vision_app"
EOF

    chmod +x build_amd.sh
    print_status "Created AMD-specific build script: ./build_amd.sh"
}

# Test ROCm installation
test_rocm() {
    print_header "Testing ROCm installation..."
    
    # Create simple HIP test program
    cat > hip_test.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cout << "HIP Error: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    
    return 0;
}
EOF

    # Compile test program
    if /opt/rocm/bin/hipcc hip_test.cpp -o hip_test 2>/dev/null; then
        print_status "HIP compilation successful"
        
        # Run test program
        if ./hip_test; then
            print_status "HIP test passed!"
        else
            print_warning "HIP test failed to run. You may need to logout/login for group changes."
        fi
        
        # Clean up
        rm -f hip_test hip_test.cpp
    else
        print_error "HIP compilation failed!"
        rm -f hip_test.cpp
        return 1
    fi
}

# Update CMakeLists.txt for better HIP support
update_cmake() {
    print_header "Updating CMakeLists.txt for HIP support..."
    
    # The CMakeLists.txt should already be updated, but let's verify HIP detection
    cat > cmake/FindHIP.cmake << 'EOF'
# FindHIP.cmake - Find HIP installation
# This module defines:
#  HIP_FOUND - True if HIP is found
#  HIP_INCLUDE_DIRS - Include directories for HIP
#  HIP_LIBRARIES - Libraries to link against
#  HIP_COMPILER - HIP compiler executable

find_path(HIP_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    PATHS /opt/rocm/include
    NO_DEFAULT_PATH
)

find_library(HIP_LIBRARIES
    NAMES hip_hcc hip_nvcc
    PATHS /opt/rocm/lib
    NO_DEFAULT_PATH
)

find_program(HIP_COMPILER
    NAMES hipcc
    PATHS /opt/rocm/bin
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIP
    REQUIRED_VARS HIP_INCLUDE_DIR HIP_LIBRARIES HIP_COMPILER
)

if(HIP_FOUND)
    set(HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIR})
    mark_as_advanced(HIP_INCLUDE_DIR HIP_LIBRARIES HIP_COMPILER)
endif()
EOF

    mkdir -p cmake
    print_status "Created cmake/FindHIP.cmake"
}

# Main function
main() {
    print_header "Starting AMD GPU setup for Computer Vision Project"
    
    # Check for AMD GPU
    detect_amd_gpu
    
    # Install ROCm
    install_rocm
    
    # Configure environment
    configure_environment
    
    # Update CMake files
    update_cmake
    
    # Test installation
    test_rocm
    
    print_status "AMD GPU setup completed successfully!"
    echo ""
    print_warning "IMPORTANT: Please logout and login again (or reboot) for group changes to take effect."
    echo ""
    print_status "Next steps:"
    echo "  1. Logout and login again"
    echo "  2. Run: source ~/.bashrc"
    echo "  3. Run: ./build_amd.sh"
    echo "  4. Run: ./build_amd/stereo_vision_app"
    echo ""
    print_status "To verify ROCm is working after reboot:"
    echo "  rocm-smi"
    echo "  hipcc --version"
}

# Run main function
main "$@"
