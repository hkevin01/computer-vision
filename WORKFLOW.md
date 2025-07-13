# Development Workflow

## 🚀 Getting Started

### Prerequisites
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10+, or macOS 11+
- **Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **Build Tools**: CMake 3.18+, Ninja (recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space for build artifacts

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd computer-vision

# Install development environment
./build_scripts/setup_dev_environment.sh

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Build and test
./run.sh --tests
```

## 🏗️ Build System

### Build Configurations

#### Standard Builds
```bash
./run.sh                    # Default build and run
./run.sh --help             # Show all options
./run.sh --clean            # Clean build from scratch
./run.sh --debug            # Debug build
./run.sh --build-only       # Build without running
```

#### GPU-Specific Builds
```bash
./run.sh --amd              # AMD/HIP GPU build
./run.sh --cpu-only         # CPU-only build (no GPU)
```

#### Development Builds
```bash
./run.sh --tests            # Build and run test suite
./run.sh --force-reconfig   # Fix CMake configuration issues
./run.sh --target core      # Build specific target
```

### Build Targets

| Target | Description | Usage |
|--------|-------------|-------|
| `stereo_vision_core` | Core library | `--target stereo_vision_core` |
| `stereo_vision_gui` | GUI library | `--target stereo_vision_gui` |
| `stereo_vision_app` | Main application | `--target stereo_vision_app` |
| `stereo_vision_app_simple` | Simplified app | `--target stereo_vision_app_simple` |
| `run_tests` | Test suite | `--tests` |

## 🧪 Testing Framework

### Running Tests

#### All Tests
```bash
./run.sh --tests           # Complete test suite
```

#### Individual Components
```bash
# Test specific components
cd test_programs
./test_camera_manager_simple
./test_gui_camera_detection
```

#### Test Categories
- **Unit Tests**: `tests/test_*.cpp` - Component isolation testing
- **Integration Tests**: `test_programs/` - Cross-component testing
- **Performance Tests**: Benchmarking and performance validation
- **GUI Tests**: User interface and interaction testing

### Test Data
- **Location**: `data/test_data/`
- **Format**: Sample images, calibration data, ground truth
- **Usage**: Automatically loaded by test programs

## 🔄 Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/awesome-feature

# Make changes
# Edit code, add tests, update documentation

# Test changes
./run.sh --tests

# Format code (automatic with pre-commit)
pre-commit run --all-files

# Commit changes
git add .
git commit -m "feat: add awesome feature"

# Push and create PR
git push origin feature/awesome-feature
```

### 2. Code Quality Checks

#### Automatic (Pre-commit)
- **Code Formatting**: clang-format for C++
- **Static Analysis**: cppcheck integration
- **Documentation**: Markdown linting
- **Security**: Secret detection

#### Manual
```bash
# Run formatting manually
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

# Run static analysis
cppcheck --enable=all src/ include/

# Generate documentation
doxygen docs/Doxyfile
```

### 3. Performance Testing

```bash
# Run benchmarks
cd archive/temp_tests
./test_benchmarking

# View results
firefox ../reports/benchmarks/benchmark_report.html
```

## 🛠️ Development Tools

### VS Code Integration

The project includes comprehensive VS Code configuration:

#### Extensions (Recommended)
- **C/C++**: IntelliSense and debugging
- **CMake Tools**: Build system integration
- **clang-format**: Code formatting
- **GitLens**: Git integration
- **Doxygen Documentation**: API docs

#### Tasks
- **Build Debug**: `Ctrl+Shift+P` → "Tasks: Run Task" → "cmake: build"
- **Run Tests**: Built-in test task
- **Format Code**: Automatic on save

#### Debug Configuration
- Pre-configured launch configurations for applications and tests
- Breakpoint debugging with full symbol information
- Memory and performance analysis tools

### Command Line Tools

#### CMake Commands
```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build --parallel $(nproc)

# Test
cd build && ctest --output-on-failure

# Install
cmake --install build --prefix /usr/local
```

#### Ninja (Faster Builds)
```bash
# Configure with Ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build with Ninja
ninja -C build
```

## 📊 Performance Monitoring

### Benchmarking

```bash
# Run performance benchmarks
./archive/temp_tests/test_benchmarking

# Results location
ls reports/benchmarks/
```

### Profiling

#### CPU Profiling
```bash
# Build with profiling
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Run with profiler
valgrind --tool=callgrind ./build/stereo_vision_app
```

#### GPU Profiling
```bash
# NVIDIA
nvprof ./build/stereo_vision_app

# AMD
rocprof ./build/stereo_vision_app
```

### Memory Analysis
```bash
# Memory leak detection
valgrind --leak-check=full ./build/stereo_vision_app

# Memory usage profiling
valgrind --tool=massif ./build/stereo_vision_app
```

## 🔧 Troubleshooting

### Common Issues

#### Build Failures
```bash
# Cache corruption
./run.sh --force-reconfig

# Missing dependencies
./build_scripts/setup_dev_environment.sh

# Permission issues
sudo chown -R $USER:$USER build/
```

#### Runtime Issues
```bash
# Library path issues
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Qt display issues
export QT_QPA_PLATFORM=xcb

# GPU driver issues
nvidia-smi  # Check NVIDIA
rocm-smi    # Check AMD
```

#### Test Failures
```bash
# Verbose test output
./run.sh --tests -V

# Individual test debugging
gdb ./test_programs/test_camera_manager_simple
```

### Debug Information

#### System Information
```bash
# System specs
./run.sh --check-env

# Build status
./run.sh --status

# Library versions
pkg-config --list-all | grep -E "(opencv|qt)"
```

#### Log Files
- **Build Logs**: `logs/cmake_build.log`
- **Runtime Logs**: `logs/build_output.log`
- **Test Logs**: Generated during test execution

## 📚 Documentation

### API Documentation
```bash
# Generate Doxygen docs
doxygen docs/Doxyfile

# View documentation
firefox docs/html/index.html
```

### Project Documentation
- **User Guide**: `documentation/README.md`
- **Technical Specs**: `docs/project_plan.md`
- **Setup Guide**: `documentation/setup/`
- **Feature Docs**: `documentation/features/`

## 🔄 CI/CD Integration

### GitHub Actions

The project uses automated CI/CD:

#### Triggers
- **Push to main/develop**: Full test suite
- **Pull Requests**: Build and test validation
- **Tags**: Release creation

#### Workflows
- **Code Quality**: Formatting, linting, security
- **Build Matrix**: Multi-platform builds
- **Performance**: Benchmark execution
- **Documentation**: API doc generation

#### Local CI Testing
```bash
# Run similar checks locally
pre-commit run --all-files
./run.sh --tests
```

## 🚀 Release Process

### Version Management
1. Update `CHANGELOG.md` with changes
2. Update version in `CMakeLists.txt`
3. Create and push tag: `git tag v2.1.0`
4. GitHub Actions creates release automatically

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] Security scan clean
- [ ] Changelog updated
- [ ] Version bumped

---

*For additional help, see [CONTRIBUTING.md](CONTRIBUTING.md) or create an issue.*
