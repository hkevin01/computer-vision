#!/bin/bash

# 🚀 PROJECT MODERNIZATION AUTOMATION SCRIPT
# This script implements the immediate action items from the modernization strategy

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting Project Modernization..."
echo "📍 Project Root: $PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_success "Created directory: $1"
    fi
}

# Function to create file with content if it doesn't exist
create_file() {
    local file_path="$1"
    local content="$2"
    
    if [ ! -f "$file_path" ]; then
        echo "$content" > "$file_path"
        print_success "Created file: $file_path"
    else
        print_warning "File already exists: $file_path"
    fi
}

# 1. Setup CI/CD Pipeline
setup_ci_cd() {
    print_status "Setting up CI/CD Pipeline..."
    
    create_dir ".github/workflows"
    create_dir ".github/ISSUE_TEMPLATE"
    create_dir ".github/PULL_REQUEST_TEMPLATE"
    
    # Main CI/CD workflow
    create_file ".github/workflows/ci.yml" "name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    name: Test on \${{ matrix.os }} with \${{ matrix.compiler }}
    runs-on: \${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        compiler: [gcc, clang]
        exclude:
          - os: windows-latest
            compiler: gcc
          - os: macos-latest
            compiler: gcc
    
    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive
    
    - name: Setup Dependencies
      run: |
        if [ \"\${{ matrix.os }}\" = \"ubuntu-latest\" ]; then
          sudo apt-get update
          sudo apt-get install -y cmake build-essential libopencv-dev libpcl-dev qt6-base-dev
        elif [ \"\${{ matrix.os }}\" = \"macos-latest\" ]; then
          brew install cmake opencv pcl qt6
        fi
      shell: bash
    
    - name: Configure CMake
      run: |
        cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
    
    - name: Build
      run: cmake --build build --config Release --parallel
    
    - name: Test
      working-directory: build
      run: ctest --output-on-failure
    
    - name: Upload Artifacts
      uses: actions/upload-artifact@v4
      with:
        name: build-artifacts-\${{ matrix.os }}-\${{ matrix.compiler }}
        path: build/

  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y clang-format clang-tidy cppcheck
    
    - name: Check Formatting
      run: |
        find src include -name '*.cpp' -o -name '*.hpp' | xargs clang-format --dry-run --Werror
    
    - name: Static Analysis
      run: |
        cppcheck --enable=all --error-exitcode=1 src/ include/
    
    - name: Clang-Tidy
      run: |
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        run-clang-tidy -p build
"
    
    # Security scanning workflow
    create_file ".github/workflows/security.yml" "name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly scan

jobs:
  security:
    name: Security Analysis
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: cpp
    
    - name: Build for Analysis
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libopencv-dev
        cmake -B build -DCMAKE_BUILD_TYPE=Release
        cmake --build build --parallel
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
"
    
    # Issue template
    create_file ".github/ISSUE_TEMPLATE/bug_report.md" "---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Environment:
- OS: [e.g. Ubuntu 22.04, Windows 11]
- Compiler: [e.g. GCC 11, MSVC 2022]
- GPU: [e.g. NVIDIA RTX 4090, AMD RX 7900XT]
- OpenCV Version: [e.g. 4.8.0]

## Additional Context
Add any other context about the problem here.
"
    
    print_success "CI/CD Pipeline setup complete"
}

# 2. Setup Code Quality Tools
setup_code_quality() {
    print_status "Setting up Code Quality Tools..."
    
    # Clang-format configuration
    create_file ".clang-format" "BasedOnStyle: Google
IndentWidth: 2
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: Empty
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
BreakBeforeBraces: Attach
IndentCaseLabels: true
SpacesBeforeTrailingComments: 2
Standard: c++17
"
    
    # Clang-tidy configuration
    create_file ".clang-tidy" "Checks: '
  -*,
  readability-*,
  modernize-*,
  performance-*,
  bugprone-*,
  cppcoreguidelines-*,
  google-*,
  misc-*,
  -modernize-use-trailing-return-type,
  -cppcoreguidelines-avoid-magic-numbers,
  -readability-magic-numbers
'
WarningsAsErrors: ''
HeaderFilterRegex: '(src|include)/.*'
FormatStyle: file
"
    
    # Pre-commit configuration
    create_file ".pre-commit-config.yaml" "repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
      - id: check-yaml
      - id: check-json

  - repo: https://github.com/pocc/pre-commit-hooks
    rev: v1.3.5
    hooks:
      - id: clang-format
        args: [--style=file]
      - id: clang-tidy
        args: [--config-file=.clang-tidy]

  - repo: https://github.com/crate-ci/typos
    rev: v1.16.23
    hooks:
      - id: typos
"
    
    # CPPCheck configuration
    create_file ".cppcheck" "--enable=all
--std=c++17
--platform=unix64
--language=c++
--inline-suppr
--suppress=missingIncludeSystem
--suppress=unmatchedSuppression
--suppress=unusedFunction
"
    
    print_success "Code Quality Tools setup complete"
}

# 3. Setup VS Code Workspace
setup_vscode_workspace() {
    print_status "Setting up VS Code Workspace..."
    
    create_dir ".vscode"
    
    # VS Code settings
    create_file ".vscode/settings.json" "{
  \"cmake.buildDirectory\": \"\${workspaceFolder}/build\",
  \"cmake.configureOnOpen\": true,
  \"C_Cpp.default.configurationProvider\": \"ms-vscode.cmake-tools\",
  \"C_Cpp.default.cppStandard\": \"c++17\",
  \"C_Cpp.default.intelliSenseMode\": \"gcc-x64\",
  \"files.associations\": {
    \"*.hpp\": \"cpp\",
    \"*.tpp\": \"cpp\"
  },
  \"editor.formatOnSave\": true,
  \"clang-format.executable\": \"clang-format\",
  \"clang-format.style\": \"file\",
  \"cmake.generator\": \"Unix Makefiles\"
}
"
    
    # VS Code tasks
    create_file ".vscode/tasks.json" "{
  \"version\": \"2.0.0\",
  \"tasks\": [
    {
      \"label\": \"Build Debug\",
      \"type\": \"shell\",
      \"command\": \"./run.sh\",
      \"args\": [\"--debug\"],
      \"group\": {
        \"kind\": \"build\",
        \"isDefault\": true
      },
      \"presentation\": {
        \"echo\": true,
        \"reveal\": \"always\",
        \"focus\": false,
        \"panel\": \"shared\"
      }
    },
    {
      \"label\": \"Build Release\",
      \"type\": \"shell\",
      \"command\": \"./run.sh\",
      \"args\": [\"--clean\"],
      \"group\": \"build\"
    },
    {
      \"label\": \"Run Tests\",
      \"type\": \"shell\",
      \"command\": \"./run.sh\",
      \"args\": [\"--tests\"],
      \"group\": \"test\"
    },
    {
      \"label\": \"Format Code\",
      \"type\": \"shell\",
      \"command\": \"find\",
      \"args\": [\"src\", \"include\", \"-name\", \"*.cpp\", \"-o\", \"-name\", \"*.hpp\", \"|\", \"xargs\", \"clang-format\", \"-i\"],
      \"group\": \"build\"
    }
  ]
}
"
    
    # VS Code launch configuration
    create_file ".vscode/launch.json" "{
  \"version\": \"0.2.0\",
  \"configurations\": [
    {
      \"name\": \"Debug Main Application\",
      \"type\": \"cppdbg\",
      \"request\": \"launch\",
      \"program\": \"\${workspaceFolder}/build/stereo_vision_app\",
      \"args\": [],
      \"stopAtEntry\": false,
      \"cwd\": \"\${workspaceFolder}\",
      \"environment\": [],
      \"externalConsole\": false,
      \"MIMode\": \"gdb\",
      \"setupCommands\": [
        {
          \"description\": \"Enable pretty-printing for gdb\",
          \"text\": \"-enable-pretty-printing\",
          \"ignoreFailures\": true
        }
      ],
      \"preLaunchTask\": \"Build Debug\"
    },
    {
      \"name\": \"Debug Tests\",
      \"type\": \"cppdbg\",
      \"request\": \"launch\",
      \"program\": \"\${workspaceFolder}/build/tests/test_core\",
      \"args\": [],
      \"stopAtEntry\": false,
      \"cwd\": \"\${workspaceFolder}\",
      \"environment\": [],
      \"externalConsole\": false,
      \"MIMode\": \"gdb\"
    }
  ]
}
"
    
    # VS Code extensions recommendations
    create_file ".vscode/extensions.json" "{
  \"recommendations\": [
    \"ms-vscode.cpptools\",
    \"ms-vscode.cmake-tools\",
    \"llvm-vs-code-extensions.vscode-clangd\",
    \"xaver.clang-format\",
    \"ms-vscode.cpptools-extension-pack\",
    \"twxs.cmake\",
    \"ms-python.python\",
    \"ms-vscode.hexeditor\",
    \"eamodio.gitlens\",
    \"github.copilot\",
    \"ms-vscode.live-server\"
  ]
}
"
    
    print_success "VS Code Workspace setup complete"
}

# 4. Setup Docker Development Environment
setup_docker_dev() {
    print_status "Setting up Docker Development Environment..."
    
    # Dockerfile for development
    create_file "Dockerfile.dev" "FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    wget \\
    curl \\
    pkg-config \\
    libopencv-dev \\
    libpcl-dev \\
    qt6-base-dev \\
    qt6-tools-dev \\
    qt6-tools-dev-tools \\
    libqt6opengl6-dev \\
    clang \\
    clang-format \\
    clang-tidy \\
    cppcheck \\
    gdb \\
    valgrind \\
    && rm -rf /var/lib/apt/lists/*

# Install modern CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \\
    echo 'deb https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \\
    apt-get update && apt-get install -y cmake

# Set working directory
WORKDIR /workspace

# Create non-root user
RUN useradd -m -s /bin/bash developer && \\
    usermod -aG sudo developer && \\
    echo 'developer ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER developer

# Copy project files
COPY --chown=developer:developer . /workspace/

# Build the project
RUN mkdir -p build && cd build && \\
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON && \\
    make -j\$(nproc)

CMD [\"/bin/bash\"]
"
    
    # Docker Compose for development
    create_file "docker-compose.dev.yml" "version: '3.8'

services:
  stereo-vision-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    container_name: stereo-vision-dev
    volumes:
      - .:/workspace
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    environment:
      - DISPLAY=\${DISPLAY}
    network_mode: host
    stdin_open: true
    tty: true
    command: /bin/bash
"
    
    # Dev container configuration for VS Code
    create_dir ".devcontainer"
    create_file ".devcontainer/devcontainer.json" "{
  \"name\": \"Stereo Vision Development\",
  \"dockerComposeFile\": \"../docker-compose.dev.yml\",
  \"service\": \"stereo-vision-dev\",
  \"workspaceFolder\": \"/workspace\",
  \"customizations\": {
    \"vscode\": {
      \"extensions\": [
        \"ms-vscode.cpptools\",
        \"ms-vscode.cmake-tools\",
        \"llvm-vs-code-extensions.vscode-clangd\",
        \"xaver.clang-format\"
      ],
      \"settings\": {
        \"cmake.buildDirectory\": \"/workspace/build\",
        \"C_Cpp.default.configurationProvider\": \"ms-vscode.cmake-tools\"
      }
    }
  },
  \"postCreateCommand\": \"echo 'Development container ready!'\",
  \"remoteUser\": \"developer\"
}
"
    
    print_success "Docker Development Environment setup complete"
}

# 5. Create build optimization scripts
create_build_scripts() {
    print_status "Creating enhanced build scripts..."
    
    create_dir "scripts"
    
    # Optimized build script
    create_file "scripts/build_optimized.sh" "#!/bin/bash
# High-performance optimized build

set -e

BUILD_DIR=\"build_optimized\"
CMAKE_ARGS=\"-DCMAKE_BUILD_TYPE=Release\"
CMAKE_ARGS+=\" -DCMAKE_CXX_FLAGS='-O3 -march=native -mtune=native'\"
CMAKE_ARGS+=\" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON\"

echo \"🚀 Building optimized version...\"

# Clean and create build directory
rm -rf \$BUILD_DIR
mkdir -p \$BUILD_DIR
cd \$BUILD_DIR

# Configure with optimizations
cmake .. \$CMAKE_ARGS

# Build with all available cores
make -j\$(nproc)

echo \"✅ Optimized build complete\"
echo \"📍 Executable: \$BUILD_DIR/stereo_vision_app\"
"
    
    # Benchmarking script
    create_file "scripts/run_benchmarks.sh" "#!/bin/bash
# Performance benchmarking suite

set -e

echo \"🏁 Running Performance Benchmarks...\"

# Build optimized version if needed
if [ ! -f \"build_optimized/stereo_vision_app\" ]; then
    echo \"Building optimized version first...\"
    ./scripts/build_optimized.sh
fi

# Run benchmarks
cd build_optimized

echo \"📊 Camera Calibration Benchmark:\"
./stereo_vision_app --benchmark --calibration --samples=100

echo \"📊 Stereo Matching Benchmark:\"
./stereo_vision_app --benchmark --stereo-matching --samples=50

echo \"📊 Point Cloud Generation Benchmark:\"
./stereo_vision_app --benchmark --point-cloud --samples=30

echo \"✅ Benchmarks complete\"
"
    
    # Documentation generation script
    create_file "scripts/generate_docs.sh" "#!/bin/bash
# Generate comprehensive documentation

set -e

echo \"📚 Generating Documentation...\"

# Install doxygen if not available
if ! command -v doxygen &> /dev/null; then
    echo \"Installing Doxygen...\"
    sudo apt-get update && sudo apt-get install -y doxygen graphviz
fi

# Create docs directory
mkdir -p docs/api

# Generate Doxygen configuration
cat > Doxyfile << EOF
PROJECT_NAME           = \"Stereo Vision 3D Point Cloud Generator\"
PROJECT_VERSION        = \"1.0.0\"
PROJECT_DESCRIPTION    = \"High-performance stereo vision application\"
INPUT                  = src include README.md
RECURSIVE              = YES
OUTPUT_DIRECTORY       = docs/api
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = YES
GENERATE_TREEVIEW      = YES
USE_MDFILE_AS_MAINPAGE = README.md
HAVE_DOT               = YES
DOT_IMAGE_FORMAT       = svg
INTERACTIVE_SVG        = YES
EOF

# Generate documentation
doxygen Doxyfile

echo \"✅ Documentation generated in docs/api/html/\"
echo \"🌐 Open docs/api/html/index.html in your browser\"
"
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_success "Enhanced build scripts created"
}

# 6. Create README for modernization
create_modernization_readme() {
    print_status "Creating modernization guide..."
    
    create_file "MODERNIZATION_GUIDE.md" "# 🚀 Project Modernization Guide

This guide helps you leverage the new modern development tools and workflows.

## 🛠️ Quick Setup

### 1. Install Dependencies
\`\`\`bash
# Install development tools
sudo apt-get install -y clang-format clang-tidy cppcheck pre-commit

# Install pre-commit hooks
pip install pre-commit
pre-commit install
\`\`\`

### 2. VS Code Setup
\`\`\`bash
# Open in VS Code
code .

# Install recommended extensions (prompted automatically)
# Or manually: Ctrl+Shift+P -> \"Extensions: Show Recommended Extensions\"
\`\`\`

### 3. Docker Development
\`\`\`bash
# Build and run development container
docker-compose -f docker-compose.dev.yml up -d

# Or use VS Code Dev Containers extension
# Ctrl+Shift+P -> \"Dev Containers: Rebuild and Reopen in Container\"
\`\`\`

## 🎯 Development Workflows

### Code Quality Checks
\`\`\`bash
# Format code
find src include -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

# Static analysis
cppcheck --config-file=.cppcheck src/ include/

# Lint code
clang-tidy -p build src/*.cpp
\`\`\`

### Performance Optimization
\`\`\`bash
# Build optimized version
./scripts/build_optimized.sh

# Run benchmarks
./scripts/run_benchmarks.sh
\`\`\`

### Documentation
\`\`\`bash
# Generate API documentation
./scripts/generate_docs.sh

# View documentation
xdg-open docs/api/html/index.html
\`\`\`

## 🔄 CI/CD Pipeline

The project now includes automated:
- ✅ Building on multiple platforms (Linux, Windows, macOS)
- ✅ Code quality checks (formatting, linting, static analysis)
- ✅ Security scanning (CodeQL)
- ✅ Test execution and coverage reporting

## 📊 Quality Metrics

Monitor these key metrics:
- **Test Coverage**: Target 90%+
- **Build Time**: <3 minutes
- **Static Analysis**: Zero high-severity issues
- **Performance**: >30 FPS stereo processing

## 🎨 Modern Features Added

1. **Professional CI/CD**: GitHub Actions with multi-platform testing
2. **Code Quality**: Automated formatting, linting, and static analysis
3. **Developer Experience**: VS Code workspace with debugging support
4. **Containerization**: Docker development environment
5. **Documentation**: Automated API documentation generation
6. **Performance**: Optimized build configurations and benchmarking

## 🚀 Next Steps

1. Explore the VS Code workspace features
2. Try the Docker development environment
3. Run the performance benchmarks
4. Review the generated API documentation
5. Contribute using the new quality tools

Your project is now equipped with professional-grade development tools! 🎉
"
    
    print_success "Modernization guide created"
}

# Main execution
main() {
    echo "🚀 PROJECT MODERNIZATION AUTOMATION"
    echo "===================================="
    
    setup_ci_cd
    echo ""
    
    setup_code_quality
    echo ""
    
    setup_vscode_workspace
    echo ""
    
    setup_docker_dev
    echo ""
    
    create_build_scripts
    echo ""
    
    create_modernization_readme
    echo ""
    
    print_success "🎉 PROJECT MODERNIZATION COMPLETE!"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "   1. Review the generated files"
    echo "   2. Install pre-commit hooks: 'pre-commit install'"
    echo "   3. Open in VS Code and install recommended extensions"
    echo "   4. Try the Docker development environment"
    echo "   5. Run './scripts/generate_docs.sh' for API documentation"
    echo ""
    echo "📖 See MODERNIZATION_GUIDE.md for detailed instructions"
}

# Run main function
main "$@"
