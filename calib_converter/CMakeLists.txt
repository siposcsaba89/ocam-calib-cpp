cmake_minimum_required(VERSION 3.0)

project(calib_converter)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)
add_executable(calib_converter calib_converter.cpp)
target_link_libraries(calib_converter ${OpenCV_LIBRARIES} Eigen3::Eigen)
