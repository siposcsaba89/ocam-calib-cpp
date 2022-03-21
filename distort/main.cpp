#include <iostream>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;

struct OcamCalibData
{
    const int32_t pd = 5;
    float poly[5];
    const int32_t ipd = 15;
    float inv_poly[15];
    float c;
    float d;
    float e;
    float cx;
    float cy;
    float iw;
    float ih;
};

bool loadOcamCalibFile(const std::string& calib_f_name, OcamCalibData& calib_data);
bool loadOcamCalibFileCPP(const std::string& calib_f_name, OcamCalibData& calib_data);
void undistortImageOcam(const OcamCalibData& cd,
    const uint8_t* img,
    int num_ch,
    const Eigen::Matrix3f& new_cam_mat_inv,
    uint8_t* o_img,
    int32_t ow,
    int32_t oh,
    int32_t onum_ch)
{
    double min_polyval = 1e100;
    double max_polyval = 0;
    for (auto j = 0; j < oh; ++j)
    {
        for (auto i = 0; i < ow; ++i)
        {
            Eigen::Vector2f p((float)i, (float)j);
            Eigen::Vector3f tmp = new_cam_mat_inv * p.homogeneous();
            p = tmp.hnormalized();

            float dist = p.norm();

            float rho = atan2f(-1, dist);
            float tmp_rho = 1;
            float polyval = 0;
            for (auto k = 0; k < cd.ipd; ++k)
            {
                float coeff = cd.inv_poly[/*ocam_model_inv.size() - 1 - */k];
                polyval += coeff * tmp_rho;
                tmp_rho *= rho;
            }
            if (polyval < min_polyval)
                min_polyval = polyval;
            if (polyval > max_polyval)
                max_polyval = polyval;
            float xx = p.x() / dist * polyval;
            float yy = p.y() / dist * polyval;

            xx = yy * cd.e + xx + cd.cx;
            yy = yy * cd.c + xx * cd.d + cd.cy;


            if (yy < (float)cd.ih && xx < (float)cd.iw && yy > 0 && xx > 0)
            {
                //val = img[int(yy) * int(cd.iw) * num_ch + int(xx) * num_ch];
                memcpy(&o_img[j * ow * onum_ch + i * onum_ch], &img[int(yy) * int(cd.iw) * num_ch + int(xx) * num_ch], onum_ch);
            }
            else
            {
                memset(&o_img[j * ow * onum_ch + i * onum_ch], 0, onum_ch);
            }
            //o_img[j * ow *onum_ch + i * onum_ch] = val;
        }
    }
    min_polyval = 0;
    max_polyval = 0;
}
void undistortImageOcam(const std::vector<float>& ocam_model_inv,
    float c,
    float d,
    float e,
    float cx,
    float cy,
    uint8_t* img,
    int32_t iw,
    int32_t ih,
    float new_cam_mat[9],
    uint8_t* o_img,
    int32_t ow,
    int32_t oh,
    float cam_rot_x)
{

    for (auto j = 0; j < oh; ++j)
    {
        for (auto i = 0; i < ow; ++i)
        {
            //float x = i - (float) ow / 2.0f;
            //float y = j - (float) oh / 2.0f;
            float x = (i - new_cam_mat[2]) / new_cam_mat[0];
            float y = (j - new_cam_mat[5]) / new_cam_mat[4];
            float z = 1;

            float alfa = cam_rot_x / 180.0f * 3.141592f;

            float co = cosf(alfa);
            float si = sinf(alfa);

            float x2 = co * x + si * z;
            z = -si * x + co * z;
            x = x2 / z;
            y /= z;
            z /= z;

            float dist = sqrtf(x * x + y * y);

            float rho = atan2f(-z, dist);
            float tmp_rho = 1;
            float polyval = 0;
            for (auto k = 0; k < ocam_model_inv.size(); ++k)
            {
                float coeff = ocam_model_inv[/*ocam_model_inv.size() - 1 - */k];
                polyval += coeff * tmp_rho;
                tmp_rho *= rho;
            }

            float xx = x / dist * polyval;
            float yy = y / dist * polyval;

            xx = yy * e + xx + cx;
            yy = yy * c + xx * d + cy;

            uint8_t val = 0;
            if (yy < (float)ih && xx < (float)iw && yy > 0 && xx > 0)
                val = img[int(yy) * iw + int(xx)];
            o_img[j * ow + i] = val;
        }
    }

}


int main()
{

    std::string image_name = "p:/calib_data/ff000451_fisheye/00000_img_1.png";
    std::string calib_name = "d:/projects/ocam-calib-cpp/build/calib_cpp_result.txt";
    OcamCalibData ocd;
    if (!loadOcamCalibFileCPP(calib_name, ocd))
        return -1;

    cv::Mat img = cv::imread(image_name);

    cv::Mat gray1, gray2, gray3, gray4;
    cv::cvtColor(img, gray1, cv::COLOR_BGR2GRAY);

    // image_name = "d:/tmp/conti_calib/front_0/1_calib00006.png";
    // img = cv::imread(image_name);
    // cv::cvtColor(img, gray2, cv::COLOR_BGR2GRAY);
    //
    // image_name = "d:/tmp/conti_calib/front_0/2_calib00006.png";
    // img = cv::imread(image_name);
    // cv::cvtColor(img, gray3, cv::COLOR_BGR2GRAY);
    //
    // image_name = "d:/tmp/conti_calib/front_0/3_calib00006.png";
    // img = cv::imread(image_name);
    // cv::cvtColor(img, gray4, cv::COLOR_BGR2GRAY);

     //cv::imshow("image", gray1);


    cv::Mat undistorted(gray1.rows, gray1.cols, CV_8UC1);

    float n_f = ocd.iw / (2 * tanf(50 / 180.0f * 3.141592f));
    Eigen::Matrix3f new_cam_mat;
    new_cam_mat <<
        n_f, 0.0f, ocd.iw / 2.0f,
        0, n_f, ocd.ih / 2.0f,
        0.0f, 0.0f, 1.0f;
    new_cam_mat = new_cam_mat * Eigen::Vector3f(0.5f, 0.5f, 1.0f).asDiagonal();
    Eigen::Matrix3f cam_mat_inv = new_cam_mat.inverse();
    undistortImageOcam(ocd, gray1.data, 1,
        cam_mat_inv, undistorted.data, gray1.cols, gray1.rows, 1);

    //undistortImageOcam(ocd, gray2.data, 1,
    //    cam_mat_inv, undistorted1.data, gray2.cols, gray2.rows, 1);
    //
    //undistortImageOcam(ocd, gray3.data, 1,
    //    cam_mat_inv, undistorted2.data, gray3.cols, gray3.rows, 1);
    //int from = 0;
    //
    cv::imshow("undistorted", undistorted);
    cv::imwrite("d:/front_178_d_ocam.png", undistorted);
    cv::waitKey(0);


}

bool loadOcamCalibFile(const std::string& calib_f_name, OcamCalibData& calib_data)
{
    std::ifstream fs(calib_f_name);
    if (!fs.is_open())
        return false;
    std::string str;
    int poly_read = 0;
    while (!fs.eof())
    {
        getline(fs, str);
        if (str.size() == 0 || str[0] == '#')
            continue;
        if (poly_read == 0)
        {
            int32_t s;
            std::stringstream ss(str);
            ss >> s;
            for (int i = 0; i < calib_data.pd; ++i)
                ss >> calib_data.poly[i];
            ++poly_read;
        }
        else if (poly_read == 1)
        {
            int32_t s;
            std::stringstream ss(str);
            ss >> s;
            for (int i = 0; i < calib_data.ipd; ++i)
                ss >> calib_data.inv_poly[i];
            ++poly_read;
        }
        else if (poly_read == 2)
        {
            std::stringstream ss(str);
            ss >> calib_data.cy;
            ss >> calib_data.cx;
            ++poly_read;
        }
        else if (poly_read == 3)
        {
            std::stringstream ss(str);
            ss >> calib_data.c;
            ss >> calib_data.d;
            ss >> calib_data.e;
            ++poly_read;
        }
        else if (poly_read == 4)
        {
            std::stringstream ss(str);
            ss >> calib_data.ih;
            ss >> calib_data.iw;
        }
    }

    return poly_read == 4;
}

bool loadOcamCalibFileCPP(const std::string& calib_f_name, OcamCalibData& calib_data)
{
    std::ifstream fs(calib_f_name);
    if (!fs.is_open())
        return false;
    std::string str;
    int poly_read = 0;
    getline(fs, str);
    while (!fs.eof() && poly_read < 4)
    {
        getline(fs, str);
        str = str.substr(str.find(':') + 1);
        if (str.size() == 0 || str[0] == '#')
            continue;
        if (poly_read == 2)
        {
            std::stringstream ss(str);
            for (int i = 0; i < calib_data.pd; ++i)
                ss >> calib_data.poly[i];
            ++poly_read;
        }
        else if (poly_read == 3)
        {
            std::stringstream ss(str);
            memset(calib_data.inv_poly, 0, sizeof(calib_data.inv_poly));
            for (int i = 0; i < calib_data.ipd; ++i)
                ss >> calib_data.inv_poly[i];
            ++poly_read;
        }
        else if (poly_read == 1)
        {
            std::stringstream ss(str);
            ss >> calib_data.cy;
            ss >> calib_data.cx;
            ++poly_read;
        }
        else if (poly_read == 0)
        {
            std::stringstream ss(str);
            ss >> calib_data.ih;
            ss >> calib_data.iw;
            ++poly_read;
        }
    }
    calib_data.c = 1.0;
    calib_data.d = 0.0;
    calib_data.e = 0.0;
    return poly_read == 4;
}
