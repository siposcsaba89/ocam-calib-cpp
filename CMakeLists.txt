cmake_minimum_required(VERSION 3.0)

project(CameraCalibration)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ceres REQUIRED)

add_executable(CameraCalibration main.cpp ocam_calibration.h ocam_calibration.cpp)

target_link_libraries(CameraCalibration ${OpenCV_LIBRARIES} ceres)
target_include_directories(CameraCalibration PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}/ceres/internal/miniglog ${EIGEN3_INCLUDE_DIRS})


add_subdirectory(distort)