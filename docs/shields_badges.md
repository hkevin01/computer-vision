# Shields.io Badge System Documentation

## Overview

[Shields.io](https://shields.io/) is a service that provides clean, consistent badges for open source projects. This document explains how badges work in our stereo vision project and how to customize them.

## What are Shields.io Badges?

Shields.io badges are small SVG images that display project metadata in a standardized format. They provide at-a-glance information about:
- Build status
- Code quality
- Dependencies
- License
- Version information
- Technology stack
- Community metrics

## Badge Anatomy

A typical shields.io badge URL follows this pattern:
```
https://img.shields.io/badge/{LABEL}-{MESSAGE}-{COLOR}
```

### Components:
- **LABEL**: Left side text (e.g., "Build")
- **MESSAGE**: Right side text (e.g., "Passing")
- **COLOR**: Badge color (hex code or predefined colors)

## Badge Categories in Our Project

### 1. Build & CI/CD Badges
```markdown
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![CMake](https://img.shields.io/badge/CMake-3.18+-blue)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)
```

### 2. Technology Stack Badges
```markdown
![C++](https://img.shields.io/badge/C++-17-blue?logo=cplusplus)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red?logo=opencv)
![Qt](https://img.shields.io/badge/Qt-6.0+-green?logo=qt)
![PCL](https://img.shields.io/badge/PCL-1.12+-orange)
```

### 3. GPU Support Badges
```markdown
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia)
![HIP](https://img.shields.io/badge/HIP-ROCm-red?logo=amd)
![OpenCL](https://img.shields.io/badge/OpenCL-3.0+-blue)
```

### 4. Platform Support Badges
```markdown
![Linux](https://img.shields.io/badge/Linux-Supported-yellow?logo=linux)
![Windows](https://img.shields.io/badge/Windows-Supported-blue?logo=windows)
![macOS](https://img.shields.io/badge/macOS-Supported-lightgrey?logo=apple)
```

### 5. License & Documentation Badges
```markdown
![License](https://img.shields.io/badge/License-MIT-blue)
![Documentation](https://img.shields.io/badge/Docs-Available-brightgreen)
![Code Style](https://img.shields.io/badge/Code%20Style-Google-yellow)
```

## Dynamic Badges

Shields.io can pull real-time data from various sources:

### GitHub Integration
```markdown
![GitHub Stars](https://img.shields.io/github/stars/username/repo)
![GitHub Forks](https://img.shields.io/github/forks/username/repo)
![GitHub Issues](https://img.shields.io/github/issues/username/repo)
![GitHub License](https://img.shields.io/github/license/username/repo)
```

### Package Managers
```markdown
![npm version](https://img.shields.io/npm/v/package-name)
![PyPI version](https://img.shields.io/pypi/v/package-name)
![Conda version](https://img.shields.io/conda/v/conda-forge/package-name)
```

### CI/CD Services
```markdown
![GitHub Actions](https://img.shields.io/github/workflow/status/username/repo/CI)
![Travis CI](https://img.shields.io/travis/username/repo)
![AppVeyor](https://img.shields.io/appveyor/ci/username/repo)
```

## Custom Badge Creation

### Method 1: Simple Text Badges
```
https://img.shields.io/badge/Stereo_Vision-3D_Point_Cloud-blue
```

### Method 2: Custom JSON Endpoint
Create a JSON endpoint that returns:
```json
{
  "schemaVersion": 1,
  "label": "build",
  "message": "passing",
  "color": "brightgreen"
}
```

Then reference it:
```
https://img.shields.io/endpoint?url=https://your-api.com/badge
```

### Method 3: Query Parameters
```
https://img.shields.io/badge/Coverage-85%25-yellow?style=flat-square&logo=codecov
```

## Badge Styles

Shields.io supports different visual styles:

### Style Options:
- `plastic` (default)
- `flat`
- `flat-square`
- `for-the-badge`
- `social`

### Examples:
```markdown
![Plastic](https://img.shields.io/badge/Style-Plastic-blue)
![Flat](https://img.shields.io/badge/Style-Flat-blue?style=flat)
![Square](https://img.shields.io/badge/Style-Square-blue?style=flat-square)
![Big](https://img.shields.io/badge/Style-Big-blue?style=for-the-badge)
```

## Color Schemes

### Predefined Colors:
- `brightgreen`, `green`, `yellowgreen`
- `yellow`, `orange`, `red`
- `lightgrey`, `blue`, `informational`
- `success`, `important`, `critical`

### Custom Colors:
Use hex codes without the `#`:
```
https://img.shields.io/badge/Custom-Color-ff69b4
```

## Logo Integration

Add logos to badges using the `logo` parameter:

### Popular Logos:
```markdown
![Docker](https://img.shields.io/badge/Docker-Supported-blue?logo=docker)
![Git](https://img.shields.io/badge/Git-VCS-orange?logo=git)
![VSCode](https://img.shields.io/badge/VSCode-Editor-blue?logo=visualstudiocode)
```

### Custom Logos:
Upload to a service and reference:
```
?logo=data:image/svg+xml;base64,{BASE64_ENCODED_SVG}
```

## Best Practices

### 1. Badge Placement
- Place most important badges first
- Group related badges together
- Don't overcrowd the README

### 2. Information Hierarchy
```markdown
<!-- Critical Info -->
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

<!-- Technical Details -->
![C++17](https://img.shields.io/badge/C++-17-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red)

<!-- Platform Support -->
![Linux](https://img.shields.io/badge/Linux-✓-yellow)
![Windows](https://img.shields.io/badge/Windows-✓-blue)
```

### 3. Maintenance
- Use dynamic badges where possible
- Update static badges regularly
- Remove outdated badges

## Advanced Features

### Multi-line Badges
```
https://img.shields.io/badge/Multi-Line%0ABadge-blue
```

### Badge Links
```markdown
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)](https://ci-service.com/build)
```

### Conditional Badges
Use GitHub Actions or other CI to show different badges based on conditions.

## Project-Specific Badge Examples

For our Stereo Vision project:

### Performance Badges
```markdown
![FPS](https://img.shields.io/badge/FPS-30+-brightgreen)
![Memory](https://img.shields.io/badge/Memory-<1GB-blue)
![Accuracy](https://img.shields.io/badge/Depth_Accuracy-99%25-brightgreen)
```

### Feature Badges
```markdown
![Stereo Vision](https://img.shields.io/badge/Stereo-Vision-purple)
![Point Cloud](https://img.shields.io/badge/Point-Cloud-orange)
![3D Reconstruction](https://img.shields.io/badge/3D-Reconstruction-red)
![Noise Reduction](https://img.shields.io/badge/Noise-Reduction-green)
```

### GPU Acceleration
```markdown
![CUDA Cores](https://img.shields.io/badge/CUDA-Accelerated-76B900)
![ROCm](https://img.shields.io/badge/ROCm-Compatible-red)
![Performance](https://img.shields.io/badge/GPU_Speedup-10x-brightgreen)
```

## Monitoring and Analytics

### Badge Click Tracking
Some services provide analytics on badge clicks to understand user engagement.

### A/B Testing
Test different badge styles and messages to see what resonates with users.

## Troubleshooting

### Common Issues:
1. **Badge not updating**: Check cache, try force refresh
2. **Special characters**: URL encode special characters
3. **Logo not showing**: Verify logo name and availability
4. **Wrong colors**: Check color name spelling or hex format

### Debugging Tools:
- Shields.io badge builder: https://shields.io/
- URL encoder for special characters
- SVG validators for custom logos

## Conclusion

Badges are a powerful way to communicate project status and capabilities at a glance. Use them strategically to enhance your project's professional appearance and provide quick access to important information.

For more advanced features and the latest updates, visit the [official Shields.io documentation](https://shields.io/).
