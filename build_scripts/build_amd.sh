#!/bin/bash

# This script configures and builds the project for AMD GPUs using HIP.

set -e # Exit immediately if a command exits with a non-zero status.

WORKSPACE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="${WORKSPACE_ROOT}/build_amd"
LOG_FILE="${WORKSPACE_ROOT}/cmake_build.log"

echo "--- AMD/HIP Build Script ---"
echo "Workspace Root: ${WORKSPACE_ROOT}"
echo "Build Directory:  ${BUILD_DIR}"
echo "Log File:         ${LOG_FILE}"

# Clear previous log file
> "${LOG_FILE}"

echo "
--- [1/3] Cleaning previous build artifacts ---" | tee -a "${LOG_FILE}"
if [ -d "${BUILD_DIR}" ]; then
    echo "Removing existing build directory: ${BUILD_DIR}" | tee -a "${LOG_FILE}"
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

echo "
--- [2/3] Running CMake Configuration ---" | tee -a "${LOG_FILE}"
# Redirect stdout and stderr to the log file and also to the console
cmake -B "${BUILD_DIR}" -S "${WORKSPACE_ROOT}" -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tee -a "${LOG_FILE}"

# Check the exit status of cmake directly
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "
--- CMake Configuration FAILED. See ${LOG_FILE} for details. ---" >&2
  exit 1
fi
echo "--- CMake Configuration Succeeded ---" | tee -a "${LOG_FILE}"


echo "
--- [3/3] Running CMake Build ---" | tee -a "${LOG_FILE}"
cmake --build "${BUILD_DIR}" -- -j$(nproc) 2>&1 | tee -a "${LOG_FILE}"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "
--- CMake Build FAILED. See ${LOG_FILE} for details. ---" >&2
  exit 1
fi
echo "--- CMake Build Succeeded ---" | tee -a "${LOG_FILE}"

echo "
--- Build Complete ---" | tee -a "${LOG_FILE}"
