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
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc calib3d)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- GUI Configuration (Qt) ---
if(BUILD_GUI)
    set(QT_LIBRARIES "")
    set(QT_VERSION_MAJOR "")
    # Force Qt5 to match system VTK/PCL installation that was compiled against Qt5
    find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL REQUIRED)
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
    add_library(stereo_vision_gui ${GUI_SOURCES} ${GUI_HEADERS})
    
    # Enable Qt's automatic MOC, UIC, and RCC
    set_target_properties(stereo_vision_gui PROPERTIES
        AUTOMOC ON
        AUTOUIC ON
        AUTORCC ON
    )
    
    target_link_libraries(stereo_vision_gui PRIVATE
        stereo_vision_core
        ${QT_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        spdlog::spdlog
    )
endif()

# --- Main Application ---
add_executable(stereo_vision_app src/main.cpp)
add_executable(stereo_vision_app_simple src/main_simple.cpp)

if(BUILD_GUI)
    target_link_libraries(stereo_vision_app PRIVATE stereo_vision_gui)
    # Link Qt5 directly to main executable
    target_link_libraries(stereo_vision_app PRIVATE Qt5::Core Qt5::Widgets)
    target_include_directories(stereo_vision_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    # Simple app
    target_link_libraries(stereo_vision_app_simple PRIVATE Qt5::Core Qt5::Widgets)
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
