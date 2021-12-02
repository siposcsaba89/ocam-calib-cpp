#include "calib_converter.h"

int main(int argc, const char* argv[])
{

    std::vector<double> poly = { -808.761790039798, 0, 0.000580253898344636, -3.42188736867185e-07, 3.00547601256e-10 };
    std::vector<double> poly_inv = { 1169.63512540441,
        623.254085841451,
        -50.1297832802301,
        105.460507398029,
        44.1417584226226,
        -11.3473206372501,
        29.9396348384398,
        5.08122505159174,
        -13.0934254922489,
        16.5033337578018,
        7.60796823512556,
        -9.83544659637847,
        -2.10464181791684,
        3.32103378094772,
        1.12117493318629 
    };
    Eigen::Vector2d principal_point{ 1922.61931008, 1078.2706176 };
    Eigen::Vector2d img_size(3840, 2160);
    Eigen::Matrix3f K_out;
    std::array<float, 5> D_out;
    float xi_out;

    calib_converter::convertOcam2Mei(poly, poly_inv, principal_point, img_size, K_out, D_out, xi_out);
    return 0;
}