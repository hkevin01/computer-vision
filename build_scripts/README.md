# Build Scripts

This directory contains various build and utility scripts for the project.

## 📁 Scripts Overview

### Core Build Scripts
- **build.sh** - Basic build script
- **build_amd.sh** - AMD/HIP specific build configuration
- **build_debug.sh** - Debug build configuration
- **clean.sh** - Clean build artifacts

### Setup Scripts
- **setup_amd_gpu.sh** - AMD GPU environment setup
- **setup_dev_environment.sh** - Development environment setup

### GUI Launch Scripts
- **launch_gui.sh** - GUI application launcher
- **test_gui.sh** - GUI testing script
- **test_gui_no_sudo.sh** - GUI testing without sudo requirements

## 🚀 Usage

### Quick Start
The main build and run script is located at the project root:
```bash
# From project root
./run.sh
```

### Alternative Build Methods
```bash
# Using scripts in this directory
cd build_scripts/

# Basic build
./build.sh

# AMD/HIP build
./build_amd.sh

# Debug build
./build_debug.sh

# Clean build artifacts
./clean.sh
```

### Environment Setup
```bash
# Setup development environment
./build_scripts/setup_dev_environment.sh

# Setup AMD GPU environment
./build_scripts/setup_amd_gpu.sh
```

### GUI Testing
```bash
# Test GUI application
./build_scripts/test_gui.sh

# Test GUI without sudo
./build_scripts/test_gui_no_sudo.sh
```

## ⚠️ Important Notes

- The primary build script is `../run.sh` at the project root
- These scripts are provided for specific use cases and testing
- Always use `../run.sh` for regular development and usage
- Some scripts may require specific environment configurations

## 🔧 Maintenance

These scripts are maintained for compatibility and specific build scenarios.
For regular usage, prefer the main `run.sh` script which includes:
- Automatic dependency detection
- Cross-platform compatibility
- Snap isolation handling
- GPU backend detection
