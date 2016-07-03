#pragma once
#include <vector>
#include <opencv2/opencv.hpp>




double calibrateCameraOcam2(const std::vector<std::vector<cv::Point3f> > & objectPoints,
    const std::vector<std::vector<cv::Point2f> > & imagePoints, const cv::Size & imageSize);

