#include <iostream>

#include <adasworks/matlib/MathLibrary.h>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace adasworks;

struct OcamCalibData
{
    const int32_t pd = 5;
    float poly[5];
    const int32_t ipd = 14;
    float inv_poly[14];
    float c;
    float d;
    float e;
    float cx;
    float cy;
    float iw;
    float ih;
};

bool loadOcamCalibFile(const std::string & calib_f_name, OcamCalibData & calib_data);
void undistortImageOcam(const OcamCalibData & cd, const uint8_t * img, int num_ch, adasworks::ml::Matrix3f & new_cam_mat_inv,
    uint8_t * o_img, int32_t ow, int32_t oh, int32_t onum_ch)
{
    double min_polyval = 1e100;
    double max_polyval = 0;
    for (auto j = 0; j < oh; ++j)
    {
        for (auto i = 0; i < ow; ++i)
        {
            ml::Vector2f p((float)i, (float)j);
            ml::Vector3f tmp = new_cam_mat_inv * p.concat(1.0f);
            tmp /= tmp.z();
            p = tmp.part_lead();

            float dist = p.norm2();

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

            xx = yy*cd.e + xx + cd.cx;
            yy = yy*cd.c + xx*cd.d + cd.cy;

            
            if (yy < (float)cd.ih && xx < (float)cd.iw && yy > 0 && xx > 0)
            {
                //val = img[int(yy) * int(cd.iw) * num_ch + int(xx) * num_ch];
                memcpy(&o_img[j * ow *onum_ch + i * onum_ch], &img[int(yy) * int(cd.iw) * num_ch + int(xx) * num_ch], onum_ch);
            }
            else
            {
                memset(&o_img[j * ow *onum_ch + i * onum_ch], 0, onum_ch);
            }
            //o_img[j * ow *onum_ch + i * onum_ch] = val;
        }
    }
    min_polyval = 0;
    max_polyval = 0;
}
void undistortImageOcam(const std::vector<float> & ocam_model_inv, float c, float d, float e, float cx, float cy, uint8_t * img,
    int32_t iw, int32_t ih, float new_cam_mat[9], uint8_t * o_img, int32_t ow, int32_t oh, float cam_rot_x)
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

            float dist = sqrtf(x*x + y*y);

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

            xx = yy*e + xx + cx;
            yy = yy*c + xx*d + cy;

            uint8_t val = 0;
            if (yy < (float)ih && xx < (float)iw && yy > 0 && xx > 0)
                val = img[int(yy) * iw + int(xx)];
            o_img[j * ow + i] = val;
        }
    }

}


class Camera
{
public:
    Camera(const adasworks::ml::Vector3f & pos = adasworks::ml::Vector3f(), 
        const adasworks::ml::Vector3f & rot = adasworks::ml::Vector3f(),
        const adasworks::ml::Vector3f & scale = adasworks::ml::Vector3f(1.0f, 1.0f, 1.0f)) : m_pos(pos), m_rotataion(rot), m_scale(scale){}
    
    void setCameraMat(const adasworks::ml::Matrix3f & cam_mat) { m_camera_mat = cam_mat; }
    void setPos(const adasworks::ml::Vector3f & pos) { m_pos = pos; }
    void setRot(const adasworks::ml::Vector3f & rot) { m_rotataion = rot; }
    void setScale(const adasworks::ml::Vector3f & scale) { m_scale = scale; }
    const adasworks::ml::Matrix4f & getProjMat() { return m_projection_mat; }
    
    void calcProjMat() {
        adasworks::ml::Matrix3f rot = ml::fromAngles_XYZ(m_rotataion);
        adasworks::ml::Matrix3f tmp = m_camera_mat * adasworks::ml::Matrix3f().DiagonalMat(m_scale) * rot;
        adasworks::ml::Vector3f pos = tmp * m_pos;
        m_projection_mat = adasworks::ml::Matrix4f(adasworks::ml::ZERO);
        m_projection_mat.setSub(adasworks::ml::ScopeS<3>(0), adasworks::ml::ScopeS<3>(0), tmp);
        m_projection_mat.setCol(3, pos.concat(1.0f));
        cout << tmp << endl;
        cout << m_projection_mat << endl;
    }

    void calcProjMatInv()
    {
        calcProjMat();
        m_proj_mat_inv = m_projection_mat;
        m_proj_mat_inv.invert();
    }
    adasworks::ml::Vector3f m_pos;
    adasworks::ml::Vector3f m_rotataion;
    adasworks::ml::Vector3f m_scale;

    adasworks::ml::Matrix3f m_camera_mat;
    adasworks::ml::Matrix4f m_projection_mat;
    adasworks::ml::Matrix4f m_proj_mat_inv;
};


int main()
{

    std::string image_name = "d:/tmp/conti_calib/front_0/0_calib00006.png";
    std::string calib_name = "d:/tmp/conti_calib/front_0/calib_results_1.txt";
    OcamCalibData ocd;
    if (!loadOcamCalibFile(calib_name, ocd))
        return -1;

    cv::Mat img = cv::imread(image_name);

    cv::Mat gray1, gray2, gray3, gray4;
    cv::cvtColor(img, gray1, cv::COLOR_BGR2GRAY);

    image_name = "d:/tmp/conti_calib/front_0/1_calib00006.png";
    img = cv::imread(image_name);
    cv::cvtColor(img, gray2, cv::COLOR_BGR2GRAY);

    image_name = "d:/tmp/conti_calib/front_0/2_calib00006.png";
    img = cv::imread(image_name);
    cv::cvtColor(img, gray3, cv::COLOR_BGR2GRAY);

    image_name = "d:/tmp/conti_calib/front_0/3_calib00006.png";
    img = cv::imread(image_name);
    cv::cvtColor(img, gray4, cv::COLOR_BGR2GRAY);

    //cv::imshow("image", gray1);


    cv::Mat undistorted(gray1.rows, gray1.cols, CV_8UC1);
    cv::Mat rotated(gray1.rows, gray1.cols, CV_8UC1);
    cv::Mat undistorted1(gray1.rows, gray1.cols, CV_8UC1);
    cv::Mat undistorted2(gray1.rows, gray1.cols, CV_8UC1);
   

   
    //camera_rot = 60.0f;
    //undistortImageOcam(ocam_mod_inv, ocam_c, ocam_d, ocam_e, ocam_cx, ocam_cy, gray.getMat(cv::ACCESS_READ).data, gray.cols, gray.rows,
    //    new_cammat.data(), undistorted.data, gray.cols, gray.rows, camera_rot);
    //cv::imwrite("d:/front_60_d_2.png", undistorted);
    float n_f = ocd.iw / (2 * tanf(50 / 180.0f * 3.141592f));
    ml::Matrix3f new_cam_mat(
    {   n_f, 0.0f, ocd.iw / 2.0f,
        0, n_f, ocd.ih / 2.0f,
        0.0f, 0.0f, 1.0f
    });
    new_cam_mat = new_cam_mat * ml::Matrix3f().DiagonalMat(ml::Vector3f(0.1f, 0.1f, 1.0f)) *
        ml::fromAngles_XYZ(ml::Vector3f({ 64 / 180.0f * 3.141592f, 5 / 180.0f * 3.141592f, 6 / 180.0f * 3.141592f }));
    ml::Matrix3f cam_mat_inv = new_cam_mat;
    cam_mat_inv.invert();
    undistortImageOcam(ocd, gray1.data, 1,
        cam_mat_inv, undistorted.data, gray1.cols, gray1.rows, 1);

    //undistortImageOcam(ocd, gray2.data, 1,
    //    cam_mat_inv, undistorted1.data, gray2.cols, gray2.rows, 1);
    //
    //undistortImageOcam(ocd, gray3.data, 1,
    //    cam_mat_inv, undistorted2.data, gray3.cols, gray3.rows, 1);
    //int from = 0;
    //
    //float scale_ipm = 0.2f;
    //Camera ipm_cam1(ml::Vector3f({ 0.0f, 1.00000f, 0.0f }), //pos
    //    ml::Vector3f({ 90 / 180.0f * 3.141592f, 90 / 180.0f * 3.141592f, 0 / 180.0f * 3.141592f }), // rot
    //    ml::Vector3f({ scale_ipm, scale_ipm, 1.0f }) // scale
    //    );
    //ipm_cam1.setCameraMat(new_cam_mat);
    //ipm_cam1.calcProjMatInv();
    //
    //Camera p_cam1(ml::Vector3f({ 0.0f, 1.0f, 0.0f }), //pos
    //    ml::Vector3f({ 25 / 180.0f * 3.141592f, -5 / 180.0f * 3.141592f, -5 / 180.0f * 3.141592f }), // rot
    //    ml::Vector3f({ 1.0f, 1.0f, 1.0f }) // scale
    //);
    //p_cam1.setCameraMat(new_cam_mat);
    //p_cam1.calcProjMatInv();
    //
    //
    //
    //for (auto j = 0; j < ocd.ih; ++j)
    //{
    //    for (int i = ocd.iw / 2; i < ocd.iw; ++i)
    //    {
    //        ml::Vector4f p((float)i, (float)j - from, 1.0f, 1.0f);
    //        p = ipm_cam1.m_proj_mat_inv * p;// world coordinates
    //        p /= p.w();
    //
    //        p = p_cam1.m_projection_mat * p;
    //        p /= p.w();
    //        p /= p.z();
    //
    //        if (p.y() >= 0 && p.x() >= 0 && p.x() < ocd.iw && p.y() < ocd.ih /*&& i < ocd.ih && j < ocd.iw*/)
    //            rotated.data[j * (int)ocd.iw + i] = undistorted.data[(int)p.y() * int(ocd.iw) + (int) p.x()];
    //    }
    //}
    //
    //
    //Camera ipm_cam2(ml::Vector3f({ 0.0f, 1.00000f, 4.0f }), //pos
    //    ml::Vector3f({ 90 / 180.0f * 3.141592f, 90 / 180.0f * 3.141592f, 0 / 180.0f * 3.141592f }), // rot
    //    ml::Vector3f({ scale_ipm, scale_ipm, 1.0f }) // scale
    //);
    //ipm_cam2.setCameraMat(new_cam_mat);
    //ipm_cam2.calcProjMatInv();
    //
    //Camera p_cam2(ml::Vector3f({ 0.0f, 1.3f, 4.0f }), //pos
    //    ml::Vector3f({ 49 / 180.0f * 3.141592f, 183.0f / 180.0f * 3.141592f, 0 / 180.0f * 3.141592f }), // rot
    //    ml::Vector3f({ 1.0f, 1.0f, 1.0f }) // scale
    //);
    //
    //p_cam2.setCameraMat(new_cam_mat);
    //p_cam2.calcProjMatInv();
    //
    //
    //for (auto j = 0; j < ocd.ih; ++j)
    //{
    //    for (int i = 0; i < ocd.iw / 2; ++i)
    //    {
    //        ml::Vector4f p((float)i, (float)j - from, 1.0f, 1.0f);
    //        p = ipm_cam2.m_proj_mat_inv * p;// world coordinates
    //        p /= p.w();
    //
    //        p = p_cam2.m_projection_mat * p;
    //        p /= p.w();
    //        p /= p.z();
    //
    //        if (p.y() >= 0 && p.x() >= 0 && p.x() < ocd.iw && p.y() < ocd.ih /*&& i < ocd.ih && j < ocd.iw*/)
    //            rotated.data[j * (int)ocd.iw + i] = undistorted1.data[(int)p.y() * int(ocd.iw) + (int)p.x()];
    //    }
    //}
    //
    //
    //
    //Camera ipm_cam3(ml::Vector3f({ -1.0f, 1.00000f, 2.0f }), //pos
    //    ml::Vector3f({ 90 / 180.0f * 3.141592f, 90 / 180.0f * 3.141592f, 0 / 180.0f * 3.141592f }), // rot
    //    ml::Vector3f({ scale_ipm, scale_ipm, 1.0f }) // scale
    //);
    //ipm_cam3.setCameraMat(new_cam_mat);
    //ipm_cam3.calcProjMatInv();
    //
    //Camera p_cam3(ml::Vector3f({ -1.0f, 1.3f, 2.0f }), //pos
    //    ml::Vector3f({ -6 / 180.0f * 3.141592f, -85.0f / 180.0f * 3.141592f, -30 / 180.0f * 3.141592f }), // rot
    //    ml::Vector3f({ 1.0f, 1.0f, 1.0f }) // scale
    //);
    //
    //p_cam3.setCameraMat(new_cam_mat);
    //p_cam3.calcProjMatInv();
    //
    //
    //for (auto j = 0;/*int(ocd.ih);*/ j < ocd.ih; ++j)
    //{
    //    for (int i = 0; i < ocd.iw / 1; ++i)
    //    {
    //        ml::Vector4f p((float)i, (float)j - from, 1.0f, 1.0f);
    //        p = ipm_cam3.m_proj_mat_inv * p;// world coordinates
    //        p /= p.w();
    //
    //        p = p_cam3.m_projection_mat * p;
    //        p /= p.w();
    //        p /= p.z();
    //
    //        if (p.y() >= 0 && p.x() >= 0 && p.x() < ocd.iw && p.y() < ocd.ih /*&& i < ocd.ih && j < ocd.iw*/)
    //            rotated.data[j * (int)ocd.iw + i] = undistorted2.data[(int)p.y() * int(ocd.iw) + (int)p.x()];
    //    }
    //}
    //
    cv::imshow("undistorted", undistorted);
    //cv::imwrite("d:/front_178_d.png", undistorted);
    cv::waitKey(0);


}

bool loadOcamCalibFile(const std::string & calib_f_name, OcamCalibData & calib_data)
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
