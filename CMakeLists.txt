cmake_minimum_required(VERSION 3.0)

project(CameraCalibration)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ceres REQUIRED)

add_executable(ocam-fisheye-calib main.cpp ocam_calibration.h ocam_calibration.cpp)

target_link_libraries(ocam-fisheye-calib ${OpenCV_LIBRARIES} ceres)
target_include_directories(ocam-fisheye-calib PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}/ceres/internal/miniglog ${EIGEN3_INCLUDE_DIRS})


add_subdirectory(distort)

add_executable(pinhole-calib pinhole-calib.cpp)
target_include_directories(pinhole-calib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pinhole-calib PRIVATE ${OpenCV_LIBRARIES})
