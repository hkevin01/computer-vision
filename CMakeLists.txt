cmake_minimum_required(VERSION 3.18)
project(StereoVisionPointCloud LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for GPU backend
option(USE_CUDA "Enable CUDA support for NVIDIA GPUs" ON)
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
find_package(PCL REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find GTK3 for GUI
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${GTK3_INCLUDE_DIRS})

# Link directories
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${GTK3_LIBRARY_DIRS})

# Add definitions
add_definitions(${PCL_DEFINITIONS})
add_compile_options(${GTK3_CFLAGS_OTHER})

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

# Create GUI library
file(GLOB_RECURSE GUI_SOURCES "src/gui/*.cpp")
add_library(stereo_vision_gui ${GUI_SOURCES})
target_link_libraries(stereo_vision_gui ${GTK3_LIBRARIES} stereo_vision_core)

# Create main executable
add_executable(stereo_vision_app src/main.cpp)
target_link_libraries(stereo_vision_app 
    stereo_vision_core 
    stereo_vision_gui 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} 
    ${GTK3_LIBRARIES}
    Threads::Threads
)

# Link GPU libraries to main executable
if(USE_CUDA)
    target_link_libraries(stereo_vision_app CUDA::cudart CUDA::cublas)
elseif(USE_HIP)
    target_link_libraries(stereo_vision_app hip::host)
endif()

# Create test executable
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
add_executable(test_stereo_vision ${TEST_SOURCES})
target_link_libraries(test_stereo_vision stereo_vision_core ${OpenCV_LIBS} ${PCL_LIBRARIES})

# Install targets
install(TARGETS stereo_vision_app DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
