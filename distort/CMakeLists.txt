cmake_minimum_required(VERSION 3.0)

project(OcamDistort)


find_package(OpenCV REQUIRED)

add_executable(OcamDistort main.cpp)

target_link_libraries(OcamDistort ${OpenCV_LIBRARIES})
target_include_directories(OcamDistort PRIVATE ${OpenCV_INCLUDE_DIRS})
