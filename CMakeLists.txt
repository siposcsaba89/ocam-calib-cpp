cmake_minimum_required(VERSION 3.0)

project(CameraCalibration)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ceres REQUIRED)

add_executable(ocam-fisheye-calib fisheye-calib.cpp ocam_calibration.h ocam_calibration.cpp)

target_link_libraries(ocam-fisheye-calib ${OpenCV_LIBRARIES} ceres)
target_include_directories(ocam-fisheye-calib PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}/ceres/internal/miniglog ${EIGEN3_INCLUDE_DIRS})


add_subdirectory(distort)
add_subdirectory(calib_converter)

add_executable(pinhole-calib pinhole-calib.cpp)
target_include_directories(pinhole-calib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pinhole-calib PRIVATE ${OpenCV_LIBRARIES})

add_executable(stereo-calib stereo-calib.cpp)
target_include_directories(stereo-calib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stereo-calib PRIVATE ${OpenCV_LIBRARIES})

add_executable(imagelist_creator imagelist_creator.cpp)
target_link_libraries(imagelist_creator PRIVATE ${OpenCV_LIBRARIES})

