cmake_minimum_required(VERSION 3.18)
project(StereoVisionPointCloud LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(C# Create GUI library (if GUI sources exist)
file(GLOB_RECURSE GUI_SOURCES "src/gui/*.cpp" "src/gui/*.h")
if(GUI_SOURCES)
    add_library(stereo_vision_gui ${GUI_SOURCES})
    target_link_libraries(stereo_vision_gui ${QT_LIBRARIES} stereo_vision_core)
    set(HAS_GUI TRUE)
else()
    message(STATUS "No GUI sources found, skipping GUI library")
    set(HAS_GUI FALSE)
endif()

# Create main executable
add_executable(stereo_vision_app src/main.cpp)
target_link_libraries(stereo_vision_app 
    stereo_vision_core 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} 
    ${QT_LIBRARIES}
    Threads::Threads
)

# Options for GPU backend
option(USE_CUDA "Enable CUDA support for NVIDIA GPUs" OFF)
option(USE_HIP "Enable HIP support for AMD GPUs" OFF)

# Auto-detect GPU backend if not specified
if(NOT USE_CUDA AND NOT USE_HIP)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(USE_CUDA ON)
        message(STATUS "Auto-detected CUDA, enabling NVIDIA GPU support")
    else()
        find_package(hip QUIET)
        if(hip_FOUND)
            set(USE_HIP ON)
            message(STATUS "Auto-detected HIP, enabling AMD GPU support")
        endif()
    endif()
endif()

# Ensure only one GPU backend is enabled
if(USE_CUDA AND USE_HIP)
    message(WARNING "Both CUDA and HIP are enabled. Prioritizing HIP for AMD GPU.")
    set(USE_CUDA OFF)
endif()

# Configure GPU backend
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DUSE_CUDA)
    message(STATUS "Building with CUDA support")
elseif(USE_HIP)
    find_package(hip REQUIRED)
    add_definitions(-DUSE_HIP)
    message(STATUS "Building with HIP support")
else()
    message(STATUS "Building with CPU-only support")
endif()

# Find required packages
find_package(OpenCV REQUIRED)

# Handle MPI configuration for VTK/PCL compatibility
if(NOT TARGET MPI::MPI_C)
    find_package(MPI QUIET)
    if(MPI_FOUND)
        message(STATUS "MPI found and configured")
    else()
        message(STATUS "MPI not found, creating dummy target for VTK compatibility")
        add_library(MPI::MPI_C INTERFACE IMPORTED)
    endif()
endif()

# Find Qt for GUI first (try Qt6 first, fallback to Qt5)
find_package(Qt6 COMPONENTS Core Widgets OpenGL QUIET)
if(Qt6_FOUND)
    message(STATUS "Using Qt6")
    set(QT_VERSION_MAJOR 6)
    # Try to find Qt6 3D components (optional)
    find_package(Qt6 COMPONENTS 3DCore 3DRender 3DInput 3DExtras QUIET)
    if(Qt6_3DCore_FOUND)
        set(QT_LIBRARIES Qt6::Core Qt6::Widgets Qt6::OpenGL Qt6::3DCore Qt6::3DRender Qt6::3DInput Qt6::3DExtras)
    else()
        set(QT_LIBRARIES Qt6::Core Qt6::Widgets Qt6::OpenGL)
        message(STATUS "Qt6 3D components not found, using basic Qt6 widgets")
    endif()
else()
    find_package(Qt5 COMPONENTS Core Widgets OpenGL QUIET)
    if(Qt5_FOUND)
        message(STATUS "Using Qt5")
        set(QT_VERSION_MAJOR 5)
        # Try to find Qt5 3D components (optional)
        find_package(Qt5 COMPONENTS 3DCore 3DRender 3DInput 3DExtras QUIET)
        if(Qt5_3DCore_FOUND)
            set(QT_LIBRARIES Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::3DCore Qt5::3DRender Qt5::3DInput Qt5::3DExtras)
        else()
            set(QT_LIBRARIES Qt5::Core Qt5::Widgets Qt5::OpenGL)
            message(STATUS "Qt5 3D components not found, using basic Qt5 widgets")
        endif()
    else()
        message(FATAL_ERROR "Neither Qt6 nor Qt5 found. Please install Qt development packages.")
    endif()
endif()

# Enable Qt MOC, UIC, and RCC
if(Qt6_FOUND)
    qt6_standard_project_setup()
else()
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
endif()

# Find other packages after Qt to avoid conflicts
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Handle PCL with VTK/MPI issues (after Qt setup to avoid conflicts)
set(VTK_LEGACY_REMOVE OFF CACHE BOOL "Remove legacy VTK")
set(VTK_USE_MPI OFF CACHE BOOL "Build VTK with MPI support")
find_package(PCL REQUIRED COMPONENTS common io features surface segmentation filters)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/gui)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Add definitions
add_definitions(${PCL_DEFINITIONS})

# Set GPU-specific flags
if(USE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
elseif(USE_HIP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")
endif()

# Create library for core functionality
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp")

# Add GPU sources based on backend
if(USE_CUDA)
    file(GLOB_RECURSE GPU_SOURCES "src/cuda/*.cu")
elseif(USE_HIP)
    file(GLOB_RECURSE GPU_SOURCES "src/cuda/*.hip.cpp")
endif()

add_library(stereo_vision_core ${CORE_SOURCES} ${GPU_SOURCES} ${UTILS_SOURCES})

# Link GPU libraries
if(USE_CUDA)
    target_link_libraries(stereo_vision_core ${OpenCV_LIBS} ${PCL_LIBRARIES} CUDA::cudart CUDA::cublas)
elseif(USE_HIP)
    target_link_libraries(stereo_vision_core ${OpenCV_LIBS} ${PCL_LIBRARIES} hip::host)
else()
    target_link_libraries(stereo_vision_core ${OpenCV_LIBS} ${PCL_LIBRARIES})
endif()

# Create GUI library (if GUI sources exist)
file(GLOB_RECURSE GUI_SOURCES "src/gui/*.cpp")
if(GUI_SOURCES)
    add_library(stereo_vision_gui ${GUI_SOURCES})
    target_link_libraries(stereo_vision_gui ${GTK3_LIBRARIES} stereo_vision_core)
    set(HAS_GUI TRUE)
else()
    message(STATUS "No GUI sources found, skipping GUI library")
    set(HAS_GUI FALSE)
endif()

# Create main executable
add_executable(stereo_vision_app src/main.cpp)
target_link_libraries(stereo_vision_app 
    stereo_vision_core 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} 
    ${GTK3_LIBRARIES}
    Threads::Threads
)

# Link GUI library if available
if(HAS_GUI)
    target_link_libraries(stereo_vision_app stereo_vision_gui)
endif()

# Link GPU libraries to main executable
if(USE_CUDA)
    target_link_libraries(stereo_vision_app CUDA::cudart CUDA::cublas)
elseif(USE_HIP)
    target_link_libraries(stereo_vision_app hip::host)
endif()

# Create test executable (if test sources exist)
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
if(TEST_SOURCES)
    add_executable(test_stereo_vision ${TEST_SOURCES})
    target_link_libraries(test_stereo_vision stereo_vision_core ${OpenCV_LIBS} ${PCL_LIBRARIES})
    message(STATUS "Test executable will be built")
else()
    message(STATUS "No test sources found, skipping test executable")
endif()

# Install targets
install(TARGETS stereo_vision_app DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
