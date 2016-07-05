#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

struct OcamCalibRes
{
	std::vector<double> ss;
	std::vector<double> ss_inv;
	double center_x;
	double center_y;
	std::vector<std::vector<double>> R;
	std::vector<std::vector<double>> T;

};


double calibrateCameraOcam2(const std::vector<std::vector<cv::Point3f> > & objectPoints,
    const std::vector<std::vector<cv::Point2f> > & imagePoints, const cv::Size & imageSize, OcamCalibRes & res);

