cmake_minimum_required(VERSION 3.16)
project(test_programs)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(TEST_SOURCES
  test_onnx_load.cpp
  test_stereo_cpu.cpp
  test_pointcloud.cpp
  test_camera_list.cpp
)

foreach(src IN LISTS TEST_SOURCES)
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE ${OpenCV_LIBS} Threads::Threads)
endforeach()
