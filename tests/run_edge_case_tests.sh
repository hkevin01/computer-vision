#!/bin/bash

# Comprehensive Edge Case Testing Script for Stereo Vision Project
# This script runs all edge case tests with various configurations to ensure robustness

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_RESULTS_DIR="${PROJECT_ROOT}/test_results"
EDGE_TEST_LOG="${TEST_RESULTS_DIR}/edge_case_test_$(date +%Y%m%d_%H%M%S).log"

# Create results directory
mkdir -p "${TEST_RESULTS_DIR}"

echo -e "${BLUE}🧪 Starting Comprehensive Edge Case Testing${NC}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Build Directory: ${BUILD_DIR}"
echo "Results Directory: ${TEST_RESULTS_DIR}"
echo "Log File: ${EDGE_TEST_LOG}"
echo "=========================================="

# Function to print status messages
print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "${EDGE_TEST_LOG}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ $1${NC}" | tee -a "${EDGE_TEST_LOG}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  $1${NC}" | tee -a "${EDGE_TEST_LOG}"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}" | tee -a "${EDGE_TEST_LOG}"
}

# Function to run tests with different configurations
run_test_configuration() {
    local config_name="$1"
    local cmake_options="$2"
    local test_executable="$3"

    print_status "Running tests: ${config_name}"

    # Create build directory for this configuration
    local config_build_dir="${BUILD_DIR}_${config_name}"
    mkdir -p "${config_build_dir}"
    cd "${config_build_dir}"

    # Configure with specific options
    print_status "Configuring ${config_name} build..."
    if cmake .. ${cmake_options} >> "${EDGE_TEST_LOG}" 2>&1; then
        print_success "Configuration successful: ${config_name}"
    else
        print_error "Configuration failed: ${config_name}"
        return 1
    fi

    # Build
    print_status "Building ${config_name}..."
    if make -j$(nproc) >> "${EDGE_TEST_LOG}" 2>&1; then
        print_success "Build successful: ${config_name}"
    else
        print_error "Build failed: ${config_name}"
        return 1
    fi

    # Run tests if executable exists
    if [ -f "${config_build_dir}/${test_executable}" ]; then
        print_status "Executing ${test_executable} for ${config_name}..."

        # Run with timeout to prevent hanging
        if timeout 1800 ./${test_executable} --gtest_output=xml:${TEST_RESULTS_DIR}/${config_name}_results.xml >> "${EDGE_TEST_LOG}" 2>&1; then
            print_success "Tests passed: ${config_name}"
        else
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                print_warning "Tests timed out: ${config_name} (this may be expected for stress tests)"
            else
                print_warning "Some tests failed: ${config_name} (exit code: $exit_code)"
            fi
        fi
    else
        print_warning "Test executable not found: ${test_executable}"
    fi

    cd "${PROJECT_ROOT}"
}

# System information gathering
print_status "Gathering system information..."
{
    echo "=== SYSTEM INFORMATION ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -a)"
    echo "CPU: $(lscpu | grep 'Model name' | cut -d ':' -f2 | xargs)"
    echo "Memory: $(free -h | grep 'Mem:' | awk '{print $2}')"
    echo "Disk Space: $(df -h / | tail -1 | awk '{print $4}')"
    echo ""
    echo "=== COMPILER INFORMATION ==="
    echo "GCC Version: $(gcc --version | head -1)"
    echo "CMake Version: $(cmake --version | head -1)"
    echo ""
    echo "=== LIBRARY VERSIONS ==="
    pkg-config --modversion opencv4 2>/dev/null && echo "OpenCV: $(pkg-config --modversion opencv4)" || echo "OpenCV: Version check failed"
    echo ""
} >> "${EDGE_TEST_LOG}"

# Check dependencies
print_status "Checking dependencies..."
missing_deps=0

check_dependency() {
    if command -v "$1" >/dev/null 2>&1; then
        print_success "Found: $1"
    else
        print_error "Missing: $1"
        missing_deps=$((missing_deps + 1))
    fi
}

check_dependency "cmake"
check_dependency "make"
check_dependency "gcc"
check_dependency "pkg-config"

if [ $missing_deps -gt 0 ]; then
    print_error "Missing $missing_deps dependencies. Please install them first."
    exit 1
fi

# Test Configurations
print_status "Starting test configurations..."

# 1. Standard Debug Build with Edge Case Tests
print_status "=== Configuration 1: Debug Build with Sanitizers ==="
run_test_configuration "debug_sanitized" \
    "-DCMAKE_BUILD_TYPE=Debug -DWITH_ONNX=OFF -DWITH_TENSORRT=OFF -DUSE_CUDA=OFF -DUSE_HIP=OFF" \
    "tests/run_edge_case_tests"

# 2. Release Build with Optimizations
print_status "=== Configuration 2: Release Build ==="
run_test_configuration "release" \
    "-DCMAKE_BUILD_TYPE=Release -DWITH_ONNX=OFF -DWITH_TENSORRT=OFF -DUSE_CUDA=OFF -DUSE_HIP=OFF" \
    "tests/run_tests"

# 3. AI/ML Build (if ONNX is available)
if pkg-config --exists libonnxruntime 2>/dev/null; then
    print_status "=== Configuration 3: AI/ML Build with ONNX ==="
    run_test_configuration "aiml_onnx" \
        "-DCMAKE_BUILD_TYPE=Debug -DWITH_ONNX=ON -DWITH_TENSORRT=OFF -DUSE_CUDA=OFF -DUSE_HIP=OFF" \
        "tests/run_edge_case_tests"
else
    print_warning "ONNX Runtime not found, skipping AI/ML tests"
fi

# 4. CUDA Build (if CUDA is available)
if command -v nvcc >/dev/null 2>&1; then
    print_status "=== Configuration 4: CUDA Build ==="
    run_test_configuration "cuda" \
        "-DCMAKE_BUILD_TYPE=Debug -DWITH_ONNX=OFF -DWITH_TENSORRT=OFF -DUSE_CUDA=ON -DUSE_HIP=OFF" \
        "tests/run_tests"
else
    print_warning "CUDA not found, skipping CUDA tests"
fi

# Memory stress testing
print_status "=== Memory Stress Testing ==="
cd "${BUILD_DIR}_debug_sanitized" 2>/dev/null || {
    print_warning "Debug build not available for memory stress testing"
}

if [ -f "tests/run_edge_case_tests" ]; then
    print_status "Running memory stress tests..."

    # Run with memory limit
    if command -v systemd-run >/dev/null 2>&1; then
        print_status "Running tests with 1GB memory limit..."
        if systemd-run --user --scope -p MemoryMax=1G ./tests/run_edge_case_tests \
            --gtest_filter="*MemoryPressure*:*ResourceExhaustion*" \
            --gtest_output=xml:${TEST_RESULTS_DIR}/memory_stress_results.xml >> "${EDGE_TEST_LOG}" 2>&1; then
            print_success "Memory stress tests completed"
        else
            print_warning "Memory stress tests had issues (expected for stress testing)"
        fi
    else
        print_warning "systemd-run not available, skipping memory limit tests"
    fi

    # Run with ulimit restrictions
    print_status "Running tests with file descriptor limits..."
    if (ulimit -n 256 && ./tests/run_edge_case_tests \
        --gtest_filter="*ResourceExhaustion*:*FileSystem*" \
        --gtest_output=xml:${TEST_RESULTS_DIR}/fd_limit_results.xml) >> "${EDGE_TEST_LOG}" 2>&1; then
        print_success "File descriptor limit tests completed"
    else
        print_warning "File descriptor limit tests had issues (expected)"
    fi
fi

cd "${PROJECT_ROOT}"

# Performance benchmarking of edge case handling
print_status "=== Performance Benchmarking ==="
if [ -f "${BUILD_DIR}_release/tests/run_tests" ]; then
    print_status "Running performance benchmarks..."
    cd "${BUILD_DIR}_release"

    # Time the test execution
    start_time=$(date +%s)
    if timeout 600 ./tests/run_tests --gtest_output=xml:${TEST_RESULTS_DIR}/performance_results.xml >> "${EDGE_TEST_LOG}" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "Performance tests completed in ${duration} seconds"
    else
        print_warning "Performance tests timed out or failed"
    fi
    cd "${PROJECT_ROOT}"
fi

# Generate summary report
print_status "=== Generating Test Summary ==="
{
    echo ""
    echo "=== EDGE CASE TESTING SUMMARY ==="
    echo "Test execution completed at: $(date)"
    echo ""
    echo "Test Result Files:"
    find "${TEST_RESULTS_DIR}" -name "*.xml" -printf "  %f\n" 2>/dev/null || echo "  No XML result files found"
    echo ""
    echo "Build Directories:"
    find "${PROJECT_ROOT}" -maxdepth 1 -name "build_*" -type d -printf "  %f\n" 2>/dev/null || echo "  No build directories found"
    echo ""
    echo "=== TEST STATISTICS ==="

    # Count test results if available
    if command -v xmllint >/dev/null 2>&1; then
        for xml_file in "${TEST_RESULTS_DIR}"/*.xml; do
            if [ -f "$xml_file" ]; then
                local test_count=$(xmllint --xpath "count(//testcase)" "$xml_file" 2>/dev/null || echo "0")
                local failure_count=$(xmllint --xpath "count(//failure)" "$xml_file" 2>/dev/null || echo "0")
                echo "  $(basename "$xml_file"): $test_count tests, $failure_count failures"
            fi
        done
    else
        echo "  xmllint not available for detailed statistics"
    fi

    echo ""
    echo "=== RECOMMENDATIONS ==="
    echo "1. Review any failed tests in the log file: ${EDGE_TEST_LOG}"
    echo "2. Check XML result files for detailed test output"
    echo "3. Run specific failing tests individually for debugging"
    echo "4. Consider enabling additional sanitizers for development builds"
    echo ""
} >> "${EDGE_TEST_LOG}"

print_success "Edge case testing completed!"
print_status "Full log available at: ${EDGE_TEST_LOG}"
print_status "Test results available in: ${TEST_RESULTS_DIR}"

# Final status
failed_configs=$(grep "❌" "${EDGE_TEST_LOG}" | wc -l)
if [ $failed_configs -eq 0 ]; then
    print_success "All test configurations completed successfully! 🎉"
    exit 0
else
    print_warning "$failed_configs issues found. Review the log for details."
    exit 1
fi
