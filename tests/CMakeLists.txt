cmake_minimum_required(VERSION 3.18)

# --- Test Project ---
project(StereoVisionTests LANGUAGES CXX)

# --- Find Dependencies ---
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Qt6 COMPONENTS Test)

# --- Include Directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${GTEST_INCLUDE_DIRS}
)

# --- Test Executable ---
add_executable(run_tests
    main.cpp
    test_core.cpp
    test_gui.cpp
)

# --- Link Libraries ---
target_link_libraries(run_tests PRIVATE
    GTest::GTest
    GTest::Main
    stereo_vision_core
    stereo_vision_gui
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Qt6::Test
)

# --- Add Test to CTest ---
include(GoogleTest)
gtest_discover_tests(run_tests)
