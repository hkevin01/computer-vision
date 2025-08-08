cmake_minimum_required(VERSION 3.18)
project(StereoVisionPointCloud LANGUAGES CXX C)

# --- Fetch Spdlog ---
include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.11.0
)
FetchContent_MakeAvailable(spdlog)

# --- Project Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# --- Options ---
option(USE_CUDA "Enable CUDA support for NVIDIA GPUs" OFF)
option(USE_HIP "Enable HIP support for AMD GPUs" OFF)
option(BUILD_GUI "Build the GUI application" ON)
option(BUILD_TESTS "Build the tests" ON)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(WITH_NEURAL_NETWORKS "Enable neural network support with ONNX Runtime" ON)
option(WITH_TENSORRT "Enable TensorRT support for NVIDIA GPUs" OFF)
option(WITH_OPENVINO "Enable Intel OpenVINO support" OFF)
option(WITH_ADVANCED_CV "Enable advanced OpenCV features (ximgproc, etc.)" ON)
option(WITH_OPENMP "Enable OpenMP support" ON)

# --- GPU Backend Detection and Configuration ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- GPU Backend Detection and Configuration ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Only auto-detect if GPU backend options are not explicitly set
if(NOT DEFINED USE_CUDA AND NOT DEFINED USE_HIP)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(USE_CUDA ON CACHE BOOL "Enable CUDA support for NVIDIA GPUs" FORCE)
        message(STATUS "Auto-detected CUDA, enabling NVIDIA GPU support.")
    else()
        find_program(ROCMINFO_EXEC rocminfo)
        if(ROCMINFO_EXEC)
             set(USE_HIP ON CACHE BOOL "Enable HIP support for AMD GPUs" FORCE)
             message(STATUS "Auto-detected ROCm, enabling AMD HIP GPU support.")
        else()
            message(STATUS "No GPU backend detected. Building for CPU only.")
        endif()
    endif()
elseif(USE_CUDA)
    message(STATUS "CUDA explicitly enabled.")
elseif(USE_HIP)
    message(STATUS "HIP explicitly enabled.")
else()
    message(STATUS "GPU backends explicitly disabled.")
endif()

if(USE_CUDA AND USE_HIP)
    message(FATAL_ERROR "Both USE_CUDA and USE_HIP are enabled. Please choose only one GPU backend.")
endif()

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89") # Common NVIDIA architectures
    add_definitions(-DUSE_CUDA)
    message(STATUS "Building with CUDA support.")
elseif(USE_HIP)
    find_package(hip REQUIRED)
    # Enable HIP language for proper kernel compilation with hipcc
    enable_language(HIP)
    add_definitions(-DUSE_HIP)
    # Set HIP compiler flags to enable CUDA compatibility
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -x hip")
    message(STATUS "Building with AMD HIP support.")
else()
    message(STATUS "Building with CPU-only support.")
endif()

# --- Find Dependencies ---
# AI/ML dependency discovery
if(WITH_NEURAL_NETWORKS)
    find_package(PkgConfig QUIET)

    # Find ONNX Runtime
    set(ONNXRUNTIME_FOUND FALSE)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime
        /opt/onnxruntime/include)
    find_library(ONNXRUNTIME_LIBRARIES onnxruntime PATHS
        /usr/lib
        /usr/local/lib
        /opt/onnxruntime/lib)
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARIES)
        set(ONNXRUNTIME_FOUND TRUE)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARIES}")
    else()
        message(WARNING "ONNX Runtime not found. Install with: sudo apt-get install libonnxruntime-dev")
    endif()

    # Find TensorRT (optional)
    if(WITH_TENSORRT)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS
            /usr/include/x86_64-linux-gnu
            /usr/local/include
            /usr/local/cuda/include)
        find_library(TENSORRT_LIBRARY nvinfer PATHS
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /usr/local/cuda/lib64)
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
            message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
        else()
            message(WARNING "TensorRT not found. Neural networks will use ONNX Runtime fallback.")
        endif()
    endif()

    # Find OpenVINO (optional)
    if(WITH_OPENVINO)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(OPENVINO QUIET openvino)
            if(OPENVINO_FOUND)
                message(STATUS "Found OpenVINO: ${OPENVINO_LIBRARIES}")
            else()
                message(WARNING "OpenVINO not found. Neural networks will use ONNX Runtime fallback.")
            endif()
        endif()
    endif()
endif()

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc calib3d stereo videoio)

# Enhanced OpenCV components for GPU acceleration
if(WITH_OPENCV_CUDA AND USE_CUDA)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc calib3d stereo videoio cudaimgproc cudastereo)
    if(OpenCV_CUDA_VERSION)
        message(STATUS "Found OpenCV with CUDA support: ${OpenCV_CUDA_VERSION}")
        add_definitions(-DUSE_OPENCV_CUDA)
    else()
        message(WARNING "OpenCV found but without CUDA support")
    endif()
endif()

# Advanced OpenCV components for enhanced computer vision
if(WITH_ADVANCED_CV)
    find_package(OpenCV QUIET COMPONENTS ximgproc photo dnn)
    if(OpenCV_ximgproc_FOUND)
        list(APPEND OpenCV_LIBS opencv_ximgproc)
        add_definitions(-DWITH_OPENCV_XIMGPROC)
        message(STATUS "OpenCV ximgproc module found - enabling advanced CV features")
    endif()
    if(OpenCV_photo_FOUND)
        list(APPEND OpenCV_LIBS opencv_photo)
        add_definitions(-DWITH_OPENCV_PHOTO)
        message(STATUS "OpenCV photo module found - enabling HDR and denoising")
    endif()
    if(OpenCV_dnn_FOUND)
        list(APPEND OpenCV_LIBS opencv_dnn)
        add_definitions(-DWITH_OPENCV_DNN)
        message(STATUS "OpenCV DNN module found - enabling neural network inference")
    endif()
endif()

# Neural Network Support with ONNX Runtime
if(WITH_NEURAL_NETWORKS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ONNXRUNTIME QUIET onnxruntime)
        if(ONNXRUNTIME_FOUND)
            add_definitions(-DWITH_ONNX)
            message(STATUS "ONNX Runtime found - enabling neural network inference")
        else()
            message(WARNING "ONNX Runtime not found. Install with: sudo apt install libonnxruntime-dev")
        endif()
    else()
        message(WARNING "PkgConfig not found - cannot detect ONNX Runtime")
    endif()
endif()

# TensorRT Support (NVIDIA only)
if(WITH_TENSORRT AND USE_CUDA)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/x86_64-linux-gnu /usr/local/cuda/include)
    find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        add_definitions(-DWITH_TENSORRT)
        message(STATUS "TensorRT found - enabling high-performance inference")
    else()
        message(WARNING "TensorRT not found. Install NVIDIA TensorRT SDK.")
    endif()
endif()

# Intel OpenVINO Support
if(WITH_OPENVINO)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENVINO QUIET openvino)
        if(OPENVINO_FOUND)
            add_definitions(-DWITH_OPENVINO)
            message(STATUS "Intel OpenVINO found - enabling optimized inference")
        else()
            message(WARNING "Intel OpenVINO not found")
        endif()
    endif()
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- GUI Configuration (Qt) ---
if(BUILD_GUI)
    set(QT_LIBRARIES "")
    set(QT_VERSION_MAJOR "")
    # Force Qt5 to match system VTK/PCL installation that was compiled against Qt5
    find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL Concurrent REQUIRED)
    set(QT_VERSION_MAJOR 5)
    list(APPEND QT_LIBRARIES Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL)

    # Try to find OpenGLWidgets component (available in Qt 5.4+)
    find_package(Qt5 COMPONENTS OpenGLWidgets QUIET)
    if(Qt5OpenGLWidgets_FOUND)
        list(APPEND QT_LIBRARIES Qt5::OpenGLWidgets)
        message(STATUS "Qt5OpenGLWidgets found and will be used.")
    else()
        message(STATUS "Qt5OpenGLWidgets not found - using Qt5::OpenGL instead.")
    endif()

    message(STATUS "Using Qt5 for GUI to match system VTK/PCL installation.")
endif()

# VTK, a dependency of PCL, may require MPI. Find it after Qt.
find_package(MPI QUIET)
if(MPI_FOUND)
    message(STATUS "MPI found for PCL/VTK compatibility.")
else()
    message(STATUS "MPI not found, continuing without it. PCL/VTK might complain.")
endif()

find_package(PCL 1.12 REQUIRED)
message(STATUS "PCL version: ${PCL_VERSION}")

# --- Find OpenMP ---
if(WITH_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        add_definitions(-DWITH_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP found - enabling parallel processing")
    else()
        message(STATUS "OpenMP not found - parallel processing will be disabled")
    endif()
endif()

# Fallback: if OpenMP was requested but not found, prevent downstream (e.g. PCL) from including <omp.h>
if(WITH_OPENMP AND NOT OpenMP_CXX_FOUND)
    add_definitions(-DPCL_NO_OPENMP)
endif()

# --- Include Directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

# Cross-platform VTK include path fix
if(UNIX AND NOT APPLE)
    # Linux: Check for VTK symlink and create if needed
    if(EXISTS "/usr/include/vtk-9.1" AND NOT EXISTS "/usr/include/vtk")
        message(STATUS "Creating VTK symlink for PCL compatibility...")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "/usr/include/vtk-9.1" "/usr/include/vtk"
                       RESULT_VARIABLE SYMLINK_RESULT)
        if(SYMLINK_RESULT EQUAL 0)
            message(STATUS "VTK symlink created successfully")
        else()
            message(WARNING "Failed to create VTK symlink. You may need to run: sudo ln -sf /usr/include/vtk-9.1 /usr/include/vtk")
        endif()
    endif()
    if(EXISTS "/usr/include/vtk")
        include_directories(/usr/include/vtk)
    endif()
elseif(WIN32)
    # Windows: VTK headers are typically found automatically by PCL
    message(STATUS "Windows detected - VTK headers should be found automatically")
elseif(APPLE)
    # macOS: VTK headers are typically found automatically by PCL
    message(STATUS "macOS detected - VTK headers should be found automatically")
endif()

# --- Source Files ---
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE GPU_SOURCES "src/gpu/*.cpp")
file(GLOB_RECURSE GUI_SOURCES "src/gui/*.cpp")
file(GLOB_RECURSE GUI_HEADERS "include/gui/*.hpp")
file(GLOB_RECURSE UTIL_SOURCES "src/utils/*.cpp")

# --- Batch Processing Sources ---
set(BATCH_SOURCES
    "src/batch_processor.cpp"
)

set(BATCH_HEADERS
    "include/batch_processor.hpp"
)

# --- AI and Neural Network Sources ---
set(AI_SOURCES
    "src/ai/neural_stereo_matcher_simple.cpp"
    "src/ai/enhanced_neural_matcher.cpp"
)

set(AI_HEADERS
    "include/ai/neural_stereo_matcher_simple.hpp"
    "include/ai/enhanced_neural_matcher.hpp"
)

# --- Multi-Camera System Sources ---
set(MULTICAM_SOURCES
    "src/multicam/multi_camera_system_simple.cpp"
)

set(MULTICAM_HEADERS
    "include/multicam/multi_camera_system_simple.hpp"
)

# --- Benchmark Sources ---
set(BENCHMARK_SOURCES
    "src/benchmark/performance_benchmark_simple.cpp"
)

set(BENCHMARK_HEADERS
    "include/benchmark/performance_benchmark_simple.hpp"
)

# --- Advanced Calibration Sources ---
set(CALIBRATION_SOURCES
    "src/calibration/eight_point_algorithm.cpp"
    "src/calibration/tsai_calibration.cpp"
    "src/calibration/advanced_calibration_manager.cpp"
)

set(CALIBRATION_HEADERS
    "include/calibration/advanced_calibration.hpp"
)

# Include modern theme and performance optimization sources
set(MODERN_GUI_SOURCES
    "src/gui/modern_theme.cpp"
    "src/gui/enhanced_image_widget.cpp"
)

# Add modern sources to GUI sources
list(APPEND GUI_SOURCES ${MODERN_GUI_SOURCES})

# Add new sources to core library
list(APPEND CORE_SOURCES ${AI_SOURCES} ${MULTICAM_SOURCES} ${BATCH_SOURCES} ${CALIBRATION_SOURCES})
list(APPEND GUI_HEADERS ${AI_HEADERS} ${MULTICAM_HEADERS} ${BATCH_HEADERS} ${CALIBRATION_HEADERS})

# Add benchmark sources (optional build)
if(BUILD_BENCHMARKS)
    list(APPEND CORE_SOURCES ${BENCHMARK_SOURCES})
    list(APPEND GUI_HEADERS ${BENCHMARK_HEADERS})
endif()

# --- Core Library (updated) ---
add_library(stereo_vision_core ${CORE_SOURCES} ${GPU_SOURCES} ${UTIL_SOURCES})

# Add include directories for new features
target_include_directories(stereo_vision_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/include/multicam
    ${CMAKE_CURRENT_SOURCE_DIR}/include/benchmark
)

# Add OpenMP flags
if(OpenMP_CXX_FOUND)
    target_compile_options(stereo_vision_core PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(stereo_vision_core PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(stereo_vision_core PRIVATE
    Threads::Threads
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    spdlog::spdlog
)

# Add AI/ML dependencies if available
if(WITH_NEURAL_NETWORKS AND ONNXRUNTIME_FOUND)
    target_link_libraries(stereo_vision_core PRIVATE ${ONNXRUNTIME_LIBRARIES})
    target_include_directories(stereo_vision_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
endif()

if(WITH_TENSORRT AND TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    target_link_libraries(stereo_vision_core PRIVATE ${TENSORRT_LIBRARY})
    target_include_directories(stereo_vision_core PRIVATE ${TENSORRT_INCLUDE_DIR})
endif()

if(WITH_OPENVINO AND OPENVINO_FOUND)
    target_link_libraries(stereo_vision_core PRIVATE ${OPENVINO_LIBRARIES})
    target_include_directories(stereo_vision_core PRIVATE ${OPENVINO_INCLUDE_DIRS})
endif()

if(USE_HIP)
    target_link_libraries(stereo_vision_core PRIVATE hip::host)
endif()

# --- GUI Library ---
if(BUILD_GUI)
    add_library(stereo_vision_gui ${GUI_SOURCES} ${GUI_HEADERS})

    # Enable Qt's automatic MOC, UIC, and RCC
    set_target_properties(stereo_vision_gui PROPERTIES
        AUTOMOC ON
        AUTOUIC ON
        AUTORCC ON
    )

    target_include_directories(stereo_vision_gui PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5Widgets_INCLUDE_DIRS}
        ${Qt5OpenGL_INCLUDE_DIRS}
        ${Qt5Concurrent_INCLUDE_DIRS}
    )

    target_link_libraries(stereo_vision_gui PRIVATE
        stereo_vision_core
        Qt5::Core
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::Concurrent
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        spdlog::spdlog
    )
endif()

# --- Main Application ---
add_executable(stereo_vision_app src/main.cpp)
add_executable(stereo_vision_app_simple src/main_simple.cpp)

# --- Live Stereo Parameter Tuning Tool ---
if(BUILD_GUI)
    add_executable(live_stereo_tuning src/live_stereo_tuning_main.cpp)
    target_link_libraries(live_stereo_tuning PRIVATE stereo_vision_gui stereo_vision_core)
    target_link_libraries(live_stereo_tuning PRIVATE Qt5::Core Qt5::Widgets)
    target_include_directories(live_stereo_tuning PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Batch processing test application
    add_executable(batch_processing_test src/batch_processing_test.cpp)
    target_link_libraries(batch_processing_test PRIVATE stereo_vision_gui stereo_vision_core)
    target_link_libraries(batch_processing_test PRIVATE Qt5::Core Qt5::Widgets Qt5::Concurrent)
    target_include_directories(batch_processing_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(batch_processing_test PROPERTIES AUTOMOC ON)
endif()

# --- Core batch processing test (no GUI dependencies) ---
add_executable(batch_core_test src/batch_core_test.cpp)
target_link_libraries(batch_core_test PRIVATE Qt5::Core)
target_include_directories(batch_core_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Advanced Calibration Test ---
add_executable(test_advanced_calibration test_programs/test_advanced_calibration.cpp)
target_link_libraries(test_advanced_calibration PRIVATE stereo_vision_core)
target_include_directories(test_advanced_calibration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(BUILD_GUI)
    target_link_libraries(stereo_vision_app PRIVATE stereo_vision_gui)
    # Link Qt5 directly to main executable
    target_link_libraries(stereo_vision_app PRIVATE Qt5::Core Qt5::Widgets)
    target_include_directories(stereo_vision_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Simple app with core library
    target_link_libraries(stereo_vision_app_simple PRIVATE stereo_vision_core Qt5::Core Qt5::Widgets)
else()
    target_link_libraries(stereo_vision_app PRIVATE stereo_vision_core)
endif()

# --- Test Configuration ---
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    # Propagate Qt includes to the test project
    if(BUILD_GUI)
        if(TARGET Qt6::Core)
            target_include_directories(run_tests PRIVATE ${Qt6_INCLUDE_DIRS})
        elseif(TARGET Qt5::Core)
            target_include_directories(run_tests PRIVATE ${Qt5_INCLUDE_DIRS})
        endif()
    endif()
endif()

# --- Installation ---
install(TARGETS stereo_vision_app DESTINATION bin)
install(DIRECTORY data/ DESTINATION share/stereo_vision_app/data)

message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")

# --- Optional Dependencies for New Features ---

# TensorRT support for neural networks
option(WITH_TENSORRT "Enable TensorRT support for neural inference" OFF)
if(WITH_TENSORRT)
    find_package(TensorRT QUIET)
    if(TensorRT_FOUND)
        target_compile_definitions(stereo_vision_core PRIVATE WITH_TENSORRT)
        target_link_libraries(stereo_vision_core PRIVATE ${TensorRT_LIBRARIES})
        target_include_directories(stereo_vision_core PRIVATE ${TensorRT_INCLUDE_DIRS})
        message(STATUS "TensorRT support enabled")
    else()
        message(WARNING "TensorRT requested but not found")
    endif()
endif()

# ONNX Runtime support
option(WITH_ONNX "Enable ONNX Runtime support for neural inference" OFF)
if(WITH_ONNX)
    find_package(OnnxRuntime QUIET)
    if(OnnxRuntime_FOUND)
        target_compile_definitions(stereo_vision_core PRIVATE WITH_ONNX)
        target_link_libraries(stereo_vision_core PRIVATE ${OnnxRuntime_LIBRARIES})
        target_include_directories(stereo_vision_core PRIVATE ${OnnxRuntime_INCLUDE_DIRS})
        message(STATUS "ONNX Runtime support enabled")
    else()
        message(WARNING "ONNX Runtime requested but not found")
    endif()
endif()

# Benchmarking support
option(BUILD_BENCHMARKS "Build performance benchmarking tools" ON)

# Package generation support
option(PACKAGE_BUILD "Configure build for packaging" OFF)
if(PACKAGE_BUILD)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# --- Existing Dependencies Section ---
