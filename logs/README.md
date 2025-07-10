# Logs Directory

This directory contains build logs and output files generated during compilation and testing.

## 📁 Log Files

### Build Logs
- **cmake_build.log** - CMake configuration and build output
- **build_output.log** - General build process output

## 📊 Log File Purposes

### cmake_build.log
- Contains CMake configuration output
- Shows dependency detection results
- Records compiler and linker information
- Useful for debugging build configuration issues

### build_output.log
- General build process output
- Compilation progress and status
- Error messages and warnings
- Performance metrics and timing

## 🔍 Using Log Files

### For Troubleshooting
```bash
# Check recent build issues
tail -50 logs/cmake_build.log

# Search for specific errors
grep -i error logs/build_output.log

# Check dependency issues
grep -i "not found" logs/cmake_build.log
```

### For Performance Analysis
```bash
# Check build timing
grep -i "time" logs/build_output.log

# Check parallel build usage
grep -i "parallel" logs/build_output.log
```

## 🧹 Maintenance

### Cleaning Old Logs
```bash
# Remove old log files
rm logs/*.log

# Or move to archive
mkdir -p logs/archive
mv logs/*.log logs/archive/
```

### Log Rotation
The build system automatically manages log files, but you can manually clean them:
```bash
# From project root
./build_scripts/clean.sh  # Also cleans logs
```

## 📝 Notes

- Log files are automatically generated during build processes
- These files help diagnose build and configuration issues
- Log files can grow large over time - periodic cleanup recommended
- Essential for debugging complex build dependencies and GPU backend detection
