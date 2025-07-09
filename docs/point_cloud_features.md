# Point Cloud Viewer Features

## 🎮 Interactive Controls

### Mouse Navigation
- **Left Click + Drag**: Rotate view around point cloud center
- **Right Click + Drag**: Pan camera position
- **Mouse Wheel**: Zoom in/out smoothly
- **Double Click**: Reset to default view

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `R` | Reset view to default |
| `1` | Front view |
| `2` | Side view |
| `3` | Top view |
| `A` | Toggle auto-rotation |
| `G` | Toggle grid display |
| `X` | Toggle coordinate axes |

## 🔇 Noise Suppression

### Available Filters
1. **Statistical Outlier Removal**
   - Removes points based on statistical analysis
   - Configurable mean K neighbors and standard deviation threshold
   - Effective for general noise reduction

2. **Voxel Grid Filtering** 
   - Downsamples point cloud using 3D grid
   - Reduces point density while preserving structure
   - Improves performance for large datasets

3. **Radius Outlier Removal**
   - Removes isolated points based on local density
   - Configurable radius and minimum neighbor count
   - Good for removing scattered noise

### Filter Parameters
```cpp
// Example usage in code
widget->enableNoiseFiltering(true);
widget->setNoiseFilterParameters(
    0.01,  // leaf_size for voxel grid
    50,    // mean_k for statistical outlier removal  
    1.0    // std_dev_thresh for statistical outlier removal
);
```

## 🎨 Visualization Modes

### Color Modes
1. **RGB Mode** (Default)
   - Shows original colors from stereo cameras
   - Best for photorealistic visualization

2. **Depth Mode**
   - Color-codes points by distance from camera
   - Blue = close, Red = far
   - Useful for depth analysis

3. **Height Mode**
   - Color-codes points by Y-coordinate (height)
   - Green = high, Red = low
   - Good for terrain visualization

4. **Intensity Mode**
   - Grayscale based on RGB brightness
   - Useful for structure analysis

### Rendering Quality
- **Fast**: Optimized for real-time interaction
- **Medium**: Balanced quality and performance
- **High**: Maximum visual quality

### Lighting Options
- **Ambient Light**: Base illumination level
- **Diffuse Light**: Main lighting component
- **Specular Light**: Reflective highlights
- **Smooth Shading**: Enhanced surface rendering

## 📊 Real-time Statistics

### Available Metrics
- **Point Count**: Total number of points
- **Depth Range**: Min/max distance values
- **Average Depth**: Mean distance from camera
- **Noise Level**: Percentage of potential outliers
- **Bounding Box**: 3D dimensions (X/Y/Z ranges)
- **Memory Usage**: Current RAM consumption

### Statistics Display
```cpp
// Get current statistics
auto stats = widget->getPointCloudStatistics();
qDebug() << "Points:" << stats.numPoints;
qDebug() << "Depth range:" << stats.minDepth << "to" << stats.maxDepth;
qDebug() << "Noise level:" << (stats.noiseLevel * 100) << "%";
```

## 💾 Export Capabilities

### Supported Formats
- **PLY**: Binary and ASCII variants
- **PCD**: Point Cloud Data format
- **XYZ**: Simple coordinate format
- **PNG/JPG**: Current view as image

### Export Options
```cpp
// Export point cloud
widget->exportPointCloud("output.ply", PLY_BINARY);

// Export current view as image
widget->exportToImage("screenshot.png");
```

## 🚀 Performance Features

### Optimization Techniques
- **Level-of-Detail**: Adaptive point density based on view distance
- **Frustum Culling**: Only render visible points
- **GPU Acceleration**: OpenGL-based rendering
- **Efficient Memory Management**: Smart pointer usage

### Performance Monitoring
- Real-time FPS display
- Memory usage tracking
- GPU utilization metrics
- Point cloud processing times

## 🔧 Advanced Configuration

### Camera Settings
```cpp
// Configure camera parameters
widget->setCameraDistance(5.0f);
widget->setCameraAngles(45.0f, 30.0f); // yaw, pitch
widget->setFieldOfView(45.0f);
```

### Visual Settings
```cpp
// Customize appearance
widget->setPointSize(2.0f);
widget->setBackgroundColor(Qt::black);
widget->setWireframeMode(false);
widget->setShowAxes(true);
widget->setShowGrid(true);
```

### Noise Filter Tuning
```cpp
// Fine-tune filtering
widget->setNoiseFilterParameters(
    0.005,  // Smaller leaf size = higher detail
    100,    // More neighbors = stronger filtering
    0.5     // Lower threshold = more aggressive filtering
);
```

## 📈 Use Cases

### Research Applications
- **3D Reconstruction**: Building detailed models from stereo images
- **Computer Vision**: Algorithm development and testing
- **Robotics**: Navigation and mapping applications

### Industrial Applications  
- **Quality Control**: Dimensional analysis and inspection
- **Reverse Engineering**: Creating CAD models from physical objects
- **Automation**: Vision-guided robotic systems

### Entertainment
- **Game Development**: 3D asset creation
- **Film Production**: Set reconstruction and VFX
- **AR/VR Content**: Immersive environment creation

## 🛠️ Implementation Notes

### Thread Safety
- All GUI operations must be performed on the main thread
- Point cloud processing can be done in background threads
- Use Qt's signal/slot mechanism for thread communication

### Memory Management
- Point clouds use shared pointers for automatic cleanup
- Large datasets automatically trigger memory optimization
- Users can monitor and control memory usage

### Error Handling
- Graceful fallback for unsupported features
- Comprehensive error reporting and logging
- Recovery from OpenGL context loss

## 📝 Future Enhancements

### Planned Features
- **Mesh Generation**: Convert point clouds to triangular meshes
- **Animation Support**: Keyframe-based camera animations
- **Multi-cloud Support**: Display multiple point clouds simultaneously
- **Cloud Comparison**: Visual diff between point clouds
- **Advanced Filtering**: Machine learning-based noise detection

### Performance Improvements
- **Compute Shaders**: GPU-accelerated filtering
- **Streaming**: Real-time point cloud streaming
- **Compression**: Efficient storage and transmission
- **Multi-threading**: Parallel processing capabilities
