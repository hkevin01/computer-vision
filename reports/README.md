# Reports Directory

This directory contains generated reports, benchmarks, and analysis results from the computer vision project.

## 📁 Directory Structure

### `benchmarks/`
Performance benchmark results and reports:

- **benchmark_report.html** - Professional HTML performance report
- **benchmark_results.csv** - Detailed CSV performance data  
- **performance_baseline.csv** - Baseline metrics for regression testing

## 📊 Benchmark Reports

### Latest Performance Results

The most recent benchmarking results show excellent performance across all components:

#### Neural Network Performance
- **StereoNet**: 274 FPS average
- **PSMNet**: 268 FPS average
- **Backend**: Automatic optimization selection
- **Quality**: Full resolution confidence maps

#### Multi-Camera Performance  
- **2 Camera Setup**: 473 FPS processing rate
- **4 Camera Setup**: 236 FPS processing rate
- **Synchronization**: Hardware/Software/Timestamp modes
- **Real-time**: 30 FPS target consistently achieved

#### Traditional Algorithms
- **StereoBM**: 268 FPS (CPU optimized)
- **StereoSGBM**: 23 FPS (quality optimized)
- **Memory Usage**: 176-720 MB depending on configuration
- **CPU Usage**: 35-62% depending on algorithm

### Report Formats

#### HTML Reports (`benchmark_report.html`)
Professional presentation with:
- System information summary
- Performance results table
- Color-coded status indicators
- Responsive design for web viewing

#### CSV Data (`benchmark_results.csv`)
Machine-readable format with:
- Test name and configuration
- Performance metrics (FPS, latency, memory)
- Success/failure status
- Error messages (if any)

#### Baseline Data (`performance_baseline.csv`)
Reference data for regression testing:
- Historical performance benchmarks
- Comparison metrics for validation
- Performance trend analysis

## 🔄 Generating Reports

### Running Benchmarks

Generate new performance reports:

```bash
# Run comprehensive benchmarks
cd archive/temp_tests
./test_benchmarking

# Generated files will be in reports/benchmarks/
ls ../reports/benchmarks/
```

### Custom Benchmarking

For custom performance testing:

```bash
# Build with benchmarking enabled
./run.sh --build-only

# Run specific algorithm benchmarks
cd build
./stereo_vision_app --benchmark --algorithm=stereobm
./stereo_vision_app --benchmark --algorithm=stereosgbm
```

## 📈 Performance Analysis

### Interpreting Results

#### FPS (Frames Per Second)
- **>200 FPS**: Excellent real-time performance
- **100-200 FPS**: Good performance for most applications
- **30-100 FPS**: Acceptable for non-critical applications
- **<30 FPS**: May require optimization

#### Memory Usage
- **<500 MB**: Efficient memory utilization
- **500-1000 MB**: Moderate memory usage
- **>1000 MB**: High memory usage, monitor for leaks

#### CPU Usage
- **<50%**: Good efficiency, leaves room for other processes
- **50-80%**: Moderate usage, acceptable for dedicated applications
- **>80%**: High usage, may impact system responsiveness

### System Requirements Impact

Performance varies by system configuration:

#### Hardware Factors
- **CPU**: Core count and clock speed affect CPU algorithms
- **GPU**: Memory and compute capability affect GPU algorithms
- **RAM**: Amount affects maximum image resolution
- **Storage**: SSD vs HDD affects data loading performance

#### Software Factors
- **OS**: Linux typically shows best performance
- **Drivers**: Updated GPU drivers improve performance
- **Dependencies**: OpenCV and Qt versions affect results

## 🔧 Troubleshooting Performance

### Common Performance Issues

#### Low FPS
```bash
# Check system resources
htop
nvidia-smi  # For NVIDIA GPUs
rocm-smi    # For AMD GPUs

# Verify GPU utilization
./run.sh --check-env
```

#### High Memory Usage
```bash
# Run with memory profiling
valgrind --tool=massif ./test_benchmarking

# Check for memory leaks
valgrind --leak-check=full ./test_benchmarking
```

#### Inconsistent Results
```bash
# Run multiple benchmark iterations
for i in {1..5}; do
    ./test_benchmarking >> performance_log.txt
done
```

## 📋 Report Maintenance

### Regular Updates
- Benchmark after major changes
- Update baseline data monthly
- Archive old reports quarterly
- Document performance regressions

### Version Control
- Include reports in version control for trend analysis
- Tag reports with software version
- Maintain changelog of performance changes

### Automation
Performance reports can be automatically generated via CI/CD:

```yaml
# In .github/workflows/ci.yml
- name: Generate Performance Report
  run: |
    ./test_benchmarking
    mv benchmark_report.html reports/benchmarks/
```

## 📝 Report Templates

### Custom Report Generation

For generating custom reports:

```cpp
// Example: Custom benchmarking code
#include "benchmark/performance_benchmark_simple.hpp"

auto benchmark = std::make_shared<PerformanceBenchmark>();
auto results = benchmark->benchmarkStereoAlgorithms();
benchmark->generateHTMLReport("custom_report.html", results);
```

### Integration with External Tools

Reports can be integrated with:
- **Grafana**: Time series performance dashboards
- **Jenkins**: CI/CD performance tracking
- **Excel/Sheets**: Spreadsheet analysis
- **Jupyter**: Python analysis notebooks

## 🔗 Related Documentation

- **[Performance Benchmarking Implementation](../archive/milestone_docs/PRIORITY2_COMPLETE.md)**
- **[Test Programs Guide](../test_programs/README.md)**
- **[Build System Documentation](../build_scripts/README.md)**
- **[Development Workflow](../WORKFLOW.md)**

---

*For questions about performance reports or benchmarking, see the development workflow documentation or create an issue.*
