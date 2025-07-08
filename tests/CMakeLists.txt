cmake_minimum_required(VERSION 3.18)

# --- Test Project ---
project(StereoVisionTests LANGUAGES CXX)

# --- Find Dependencies ---
# Try to find GTest using the system installation
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # Ubuntu/Debian systems install GTest sources that need to be compiled
    # Check if system GTest is available
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h PATHS /usr/include)
    find_library(GTEST_LIBRARY gtest PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
    find_library(GTEST_MAIN_LIBRARY gtest_main PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
    
    if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY AND GTEST_MAIN_LIBRARY)
        message(STATUS "Found system GTest libraries")
        add_library(GTest::GTest STATIC IMPORTED)
        set_target_properties(GTest::GTest PROPERTIES
            IMPORTED_LOCATION ${GTEST_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
        )
        
        add_library(GTest::Main STATIC IMPORTED)
        set_target_properties(GTest::Main PROPERTIES
            IMPORTED_LOCATION ${GTEST_MAIN_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
        )
        
        set(GTest_FOUND TRUE)
        set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
        message(STATUS "Using system GTest installation")
    else()
        message(STATUS "GTest not found on system, downloading and building...")
        include(FetchContent)
        
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        release-1.12.1
        )
        
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        
        FetchContent_MakeAvailable(googletest)
        
        # Create aliases for compatibility
        if(NOT TARGET GTest::GTest)
            add_library(GTest::GTest ALIAS gtest)
        endif()
        if(NOT TARGET GTest::Main)
            add_library(GTest::Main ALIAS gtest_main)
        endif()
        
        set(GTEST_INCLUDE_DIRS ${googletest_SOURCE_DIR}/googletest/include)
        message(STATUS "GTest downloaded and configured successfully")
    endif()
endif()

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

# Use the same Qt version as the main project
# Check which Qt version the parent project is actually using by looking at targets
if(TARGET Qt6::Core)
    message(STATUS "Detected Qt6 in parent project, using Qt6 for tests")
    find_package(Qt6 COMPONENTS Test REQUIRED)
    set(QT_TEST_LIBRARIES Qt6::Test)
elseif(TARGET Qt5::Core)
    message(STATUS "Detected Qt5 in parent project, using Qt5 for tests")
    find_package(Qt5 COMPONENTS Test REQUIRED)
    set(QT_TEST_LIBRARIES Qt5::Test)
else()
    # Fallback: Try Qt5 first (more common on Ubuntu), then Qt6
    find_package(Qt5 COMPONENTS Test QUIET)
    if(Qt5_FOUND)
        set(QT_TEST_LIBRARIES Qt5::Test)
        message(STATUS "Using Qt5 for tests (fallback)")
    else()
        find_package(Qt6 COMPONENTS Test REQUIRED)
        set(QT_TEST_LIBRARIES Qt6::Test)
        message(STATUS "Using Qt6 for tests (fallback)")
    endif()
endif()

# --- Include Directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${GTEST_INCLUDE_DIRS}
)

# --- Test Executable ---
# Create test sources list
set(TEST_SOURCES
    main.cpp
    test_core.cpp
)

# Add GUI tests only if GUI is enabled
if(BUILD_GUI)
    list(APPEND TEST_SOURCES test_gui.cpp)
endif()

add_executable(run_tests ${TEST_SOURCES})

# --- Link Libraries ---
set(TEST_LIBRARIES
    GTest::GTest
    GTest::Main
    stereo_vision_core
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

# Add GUI libraries only if GUI is enabled
if(BUILD_GUI)
    list(APPEND TEST_LIBRARIES stereo_vision_gui ${QT_TEST_LIBRARIES})
endif()

target_link_libraries(run_tests PRIVATE ${TEST_LIBRARIES})

# --- Add Test to CTest ---
include(GoogleTest)
gtest_discover_tests(run_tests)
