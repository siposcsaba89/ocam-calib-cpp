#pragma once
#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <opencv2/core.hpp>


namespace calib_converter
{

double convertOcam2Mei(const std::vector<double> & poly,
    const std::vector<double> & poly_inv,
    const Eigen::Vector2d & principal_point,
    const Eigen::Vector2d & img_size,
    Eigen::Matrix3f & K_out,
    std::array<float, 5> & D_out,
    float& xi_out);
}

