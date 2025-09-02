# Contributing to Computer Vision 3D Point Cloud Generator

We welcome contributions to this project! This document provides guidelines for contributing.

## 🚀 Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/computer-vision.git
   cd computer-vision
   ```
3. **Set up development environment**:
   ```bash
   ./build_scripts/setup_dev_environment.sh
   ```
4. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 🛠️ Development Workflow

### Setting Up Your Environment

1. **System Requirements**:
   - Ubuntu 20.04+ / Windows 10+ / macOS 11+
   - GCC 9+ / Clang 10+ / MSVC 2019+
   - CMake 3.18+
   - 8GB RAM minimum

2. **Build Dependencies**:
   ```bash
   # Install core dependencies
   sudo apt-get install libopencv-dev qtbase5-dev cmake ninja-build

   # Optional GPU support
   # NVIDIA: Install CUDA Toolkit 11.0+
   # AMD: Install ROCm 5.0+
   ```

3. **Building the Project**:
   ```bash
   ./run.sh --help          # See all build options
   ./run.sh --clean         # Clean build
   ./run.sh --tests         # Build and run tests
   ```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow coding standards**:
   - Use C++17 features and modern best practices
   - Follow the existing code style (enforced by clang-format)
   - Write comprehensive tests for new features
   - Document public APIs with Doxygen comments

3. **Test your changes**:
   ```bash
   ./run.sh --tests                    # Run all tests
   ./test_programs/test_specific       # Run specific tests
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

### Code Style Guidelines

- **C++ Standard**: C++17
- **Formatting**: Automatic via clang-format (see `.clang-format`)
- **Naming Conventions**:
  - Classes: `PascalCase` (e.g., `StereoMatcher`)
  - Functions/methods: `camelCase` (e.g., `computeDisparity`)
  - Variables: `snake_case` (e.g., `image_width`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_DISPARITY`)
  - Files: `snake_case` (e.g., `stereo_matcher.hpp`)

- **Documentation**:
  ```cpp
  /**
   * @brief Computes disparity map from stereo image pair
   * @param left_image Left camera image
   * @param right_image Right camera image
   * @return Disparity map as CV_32F matrix
   */
  cv::Mat computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image);
  ```

## 🧪 Testing Guidelines

### Writing Tests

1. **Unit Tests**: Test individual components in isolation
   ```cpp
   TEST(StereoMatcherTest, BasicDisparity) {
       StereoMatcher matcher;
       cv::Mat disparity = matcher.computeDisparity(left_img, right_img);
       EXPECT_FALSE(disparity.empty());
   }
   ```

2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical algorithms
4. **GUI Tests**: Test user interface components

### Test Structure
- Place tests in appropriate `tests/` subdirectories
- Use GoogleTest framework
- Follow naming convention: `test_*.cpp`
- Include test data in `data/test_data/`

## 📝 Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(stereo): add neural network stereo matching
fix(gui): resolve camera preview memory leak
docs(readme): update installation instructions
test(calibration): add stereo calibration unit tests
```

## 🔍 Code Review Process

1. **Create Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots for UI changes
   - Performance impact notes

2. **Required Checks**:
   - All CI tests pass
   - Code coverage maintained/improved
   - Documentation updated
   - No breaking changes (or properly noted)

3. **Review Criteria**:
   - Code quality and style
   - Test coverage
   - Performance impact
   - API design consistency

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**:
   - OS and version
   - Compiler and version
   - CMake version
   - GPU type (if applicable)

2. **Reproduction Steps**:
   - Minimal code example
   - Input data (if applicable)
   - Expected vs. actual behavior

3. **Additional Context**:
   - Error messages/logs
   - Screenshots (for GUI issues)
   - Performance measurements

## ✨ Feature Requests

For new features:

1. **Check existing issues** for duplicates
2. **Describe the use case** clearly
3. **Propose implementation approach** (optional)
4. **Consider backward compatibility**

## 🏗️ Architecture Guidelines

### Core Principles
- **Modularity**: Components should have clear interfaces
- **Performance**: Optimize for real-time processing
- **Extensibility**: Easy to add new algorithms
- **Cross-platform**: Support Linux, Windows, macOS

### Adding New Components

1. **Create header in `include/`** with clean interface
2. **Implement in `src/`** with proper error handling
3. **Add comprehensive tests** in `tests/`
4. **Update documentation** in `docs/`
5. **Add example usage** in `test_programs/`

### GPU Code Guidelines
- Support both CUDA (NVIDIA) and HIP (AMD)
- Graceful fallback to CPU implementations
- Memory management with RAII
- Error checking for all GPU operations

## 📚 Documentation

### Types of Documentation
- **API Documentation**: Doxygen comments in headers
- **User Guides**: Markdown files in `documentation/`
- **Technical Specs**: Detailed docs in `docs/`
- **Examples**: Working code in `test_programs/`

### Writing Guidelines
- Use clear, concise language
- Include code examples
- Update when making changes
- Link between related documents

## 🚨 Security

### Reporting Security Issues
- **Do not** open public issues for security vulnerabilities
- Email security@project.com with details
- Allow time for fix before public disclosure

### Security Best Practices
- Validate all input data
- Use safe memory operations
- Avoid hardcoded credentials
- Regular dependency updates

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Community

### Getting Help
- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Documentation**: Check docs first

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn
- Focus on technical merit

---

Thank you for contributing to making computer vision more accessible and powerful! 🎉
