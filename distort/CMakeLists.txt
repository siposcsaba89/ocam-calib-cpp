cmake_minimum_required(VERSION 3.0)

project(ocam_distort)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(ocam_distort main.cpp)

target_link_libraries(ocam_distort ${OpenCV_LIBRARIES} Eigen3::Eigen)
target_include_directories(ocam_distort PRIVATE ${OpenCV_INCLUDE_DIRS})
