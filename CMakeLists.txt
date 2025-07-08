cmake_minimum_required(VERSION 3.18)
project(StereoVisionPointCloud LANGUAGES CXX C)

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

# --- GPU Backend Detection and Configuration ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT USE_CUDA AND NOT USE_HIP)
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
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc calib3d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- GUI Configuration (Qt) ---
if(BUILD_GUI)
    find_package(Qt6 COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets QUIET)
    if(Qt6_FOUND)
        set(QT_VERSION_MAJOR 6)
        set(QT_LIBRARIES Qt6::Core Qt6::Gui Qt6::Widgets Qt6::OpenGL Qt6::OpenGLWidgets)
        message(STATUS "Using Qt6 for GUI.")
    else()
        find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets REQUIRED)
        set(QT_VERSION_MAJOR 5)
        set(QT_LIBRARIES Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL Qt5::OpenGLWidgets)
        message(STATUS "Qt6 not found. Using Qt5 for GUI.")
    endif()
    qt_standard_project_setup()
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
file(GLOB_RECURSE UTIL_SOURCES "src/utils/*.cpp")

# --- Core Library ---
add_library(stereo_vision_core ${CORE_SOURCES} ${GPU_SOURCES} ${UTIL_SOURCES})
target_link_libraries(stereo_vision_core PRIVATE
    Threads::Threads
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)
if(USE_HIP)
    target_link_libraries(stereo_vision_core PRIVATE hip::host)
endif()

# --- GUI Library ---
if(BUILD_GUI)
    add_library(stereo_vision_gui ${GUI_SOURCES})
    target_link_libraries(stereo_vision_gui PRIVATE
        stereo_vision_core
        ${QT_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )
endif()

# --- Main Application ---
add_executable(stereo_vision_app src/main.cpp)
if(BUILD_GUI)
    target_link_libraries(stereo_vision_app PRIVATE stereo_vision_gui)
else()
    target_link_libraries(stereo_vision_app PRIVATE stereo_vision_core)
endif()

# --- Installation ---
install(TARGETS stereo_vision_app DESTINATION bin)
install(DIRECTORY data/ DESTINATION share/stereo_vision_app/data)

message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")
