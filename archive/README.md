# Archive Directory

This directory contains historical documentation and completed development artifacts that have been moved from the root directory to maintain a clean project structure.

## 📁 Directory Structure

### `milestone_docs/`
Contains documentation from completed project milestones and features:

- **ADVANCED_FEATURES_COMPLETE.md** - AI calibration and live processing completion
- **BUILD_SUCCESS*.md** - Build system success documentation
- **CALIBRATION_WIZARD_READY.md** - Camera calibration wizard completion
- **CAMERA_*.md** - Camera detection and integration documentation
- **FINAL_*.md** - Project completion summaries
- **NEXT_STEPS_*.md** - Development roadmap documents
- **PRIORITY2_COMPLETE.md** - Priority 2 features completion summary
- **PROJECT_*.md** - Project analysis and improvement documentation
- **SETUP_COMPLETE.md** - Environment setup completion
- **WEBCAM_*.md** - Webcam implementation documentation
- **WINDOWS_11_UPGRADE_COMPLETE.md** - Windows 11 theme upgrade

### `temp_tests/`
Contains completed Priority 2 feature test implementations:

- **test_benchmarking*** - Performance benchmarking test programs
- **test_multicamera*** - Multi-camera system test programs  
- **test_neural_network*** - Neural network implementation tests
- **test_priority2_*.cpp** - Priority 2 feature integration tests
- **priority2_summary*** - Priority 2 completion summary programs

## 📚 Historical Context

### Priority 2 Features (Completed July 2025)
The Priority 2 features were successfully implemented and tested:

1. **Neural Network Stereo Matching** - TensorRT/ONNX backends with adaptive optimization
2. **Multi-Camera Support** - Synchronized capture and real-time processing
3. **Professional Installers** - Cross-platform packaging framework
4. **Enhanced Performance Benchmarking** - Comprehensive testing with reports

**Performance Results:**
- Neural Networks: 274 FPS (StereoNet), 268 FPS (PSMNet)
- Multi-Camera: 473 FPS (2 cameras), 236 FPS (4 cameras)
- Complete test validation with generated reports

### Documentation Evolution
The project documentation evolved from scattered milestone files to an organized structure:

**Before:** 21+ documentation files in root directory
**After:** Organized structure with:
- `documentation/` - User and developer guides
- `docs/` - Technical documentation
- `archive/milestone_docs/` - Historical milestones

## 🔍 Using Archived Content

### Accessing Test Programs
If you need to run the Priority 2 test programs:

```bash
cd archive/temp_tests

# Neural network tests
./test_neural_network

# Multi-camera tests  
./test_multicamera

# Benchmarking tests
./test_benchmarking

# Priority 2 summary
./priority2_summary
```

### Reviewing Historical Documentation
For understanding project evolution:

```bash
# View completion summaries
cat archive/milestone_docs/PRIORITY2_COMPLETE.md
cat archive/milestone_docs/FINAL_SUCCESS.md

# Review feature implementations
cat archive/milestone_docs/ADVANCED_FEATURES_COMPLETE.md
cat archive/milestone_docs/CAMERA_FIXES_COMPLETE.md
```

### Performance Reports
Historical benchmark data is available in:
- `reports/benchmarks/` - Current structured reports
- Archive test programs can regenerate performance data

## 🚀 Current Development

For current development, use the modern project structure:

- **Tests**: Use `tests/` and `test_programs/` directories
- **Documentation**: Refer to `documentation/` and `docs/`
- **Build**: Use `./run.sh` with modern build options
- **Reports**: Check `reports/benchmarks/` for current data

## 📝 Maintenance

### When to Archive
Files are moved to archive when:
- Feature development is complete
- Documentation becomes historical
- Files clutter the root directory
- Modern equivalents exist

### Archive Policy
- **Keep**: Important historical context and achievements
- **Document**: Clear explanation of what was moved and why
- **Reference**: Maintain links to archived content where relevant
- **Access**: Ensure archived content remains accessible

## 🔗 Related Documentation

- **[Project Cleanup Summary](../docs/DIRECTORY_CLEANUP_SUMMARY.md)** - Details of the cleanup process
- **[Modernization Strategy](../PROJECT_MODERNIZATION_STRATEGY.md)** - Overall project modernization plan
- **[Current Documentation](../documentation/README.md)** - Modern documentation index
- **[Workflow Guide](../WORKFLOW.md)** - Current development workflow

---

*This archive preserves the project's development history while maintaining a clean, modern project structure.*
