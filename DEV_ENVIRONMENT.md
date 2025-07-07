# Development Environment Setup

## Compiler Configuration
- **Selected Compiler**: Clang
- **Compiler Path**: /usr/bin/clang++
- **C++ Standard**: C++17
- **CUDA Available**: false

## Build System
- **CMake Version**: 3.28.3
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
2. Run: `./build.sh` to build the project
3. Run: `./build/stereo_vision_app --help` to see usage options

## Dependencies Status
❌ CUDA: Not available
✅ OpenCV: 
✅ PCL: Available
✅ GTK3: 
