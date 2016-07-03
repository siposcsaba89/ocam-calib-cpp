cmake_minimum_required(VERSION 3.0)

project(CameraCalibration)


find_package(OpenCV REQUIRED)

add_executable(CameraCalibration main.cpp ocam_calibration.h ocam_calibration.cpp)

target_link_libraries(CameraCalibration ${OpenCV_LIBRARIES})
target_include_directories(CameraCalibration PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/awmatlib/include/eigen3/)


add_subdirectory(distort)