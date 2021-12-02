#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/omnidir.hpp>


/**
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */

Eigen::Vector2d WorldToPlane(const Eigen::Vector3d& P, const std::vector<double>& inv_poly,
    const Eigen::Vector2d& pp)
{
    double norm = std::sqrt(P[0] * P[0] + P[1] * P[1]);
    double theta = std::atan2(-P[2], norm);
    double rho = 0.0;
    double theta_i = 1.0;

    for (int i = 0; i < (int)inv_poly.size(); i++)
    {
        rho += theta_i * inv_poly[i];
        theta_i *= theta;
    }

    double invNorm = 1.0 / norm;
    Eigen::Vector2d xn(
        P[0] * invNorm * rho,
        P[1] * invNorm * rho
    );

    return Eigen::Vector2d(
        xn[0] + pp.x(),
        xn[1] + pp.y());
}


double convertOcam2Mei(const std::vector<double> & poly,
    const std::vector<double> & poly_inv,
    const Eigen::Vector2d & principal_point,
    const Eigen::Vector2d & img_size,
    Eigen::Matrix3f & K_out,
    std::array<float, 5> & D_out,
    float& xi_out)
{
    int cx = 15;
    int cy = 10;
    double sx = 0.15;
    double sy = 0.15;
    std::vector<cv::Point3f> obj_pts;
    for (int j = 0; j < cy; ++j)
    {
        for (int i = 0; i < cx; ++i)
        {
            obj_pts.push_back({ float(i * sx), float(j * sy) , 0.0f });
        }
    }

    int num_table_count = 100;
    std::vector<std::vector<cv::Point3f>> object_pts;
    std::vector<std::vector<cv::Point2f>> img_pts;
    int num_total_pts = 0;
    //cv::Mat tmp_img(img_size.y(), img_size.x(), CV_8UC1);
    std::vector<Eigen::Vector3d> orig_ts;
    std::vector<Eigen::Matrix3d> orig_R;
    std::vector<Eigen::Vector3d> orig_euler;
    for (int i = 0; i < num_table_count; ++i)
    {
        //tmp_img.setTo(0);

        std::vector<cv::Point2f> imgpts;
        std::vector<cv::Point3f> objpts;
        ///generate rotation translation
        double rx = (rand() / (double)RAND_MAX - 0.5) * 3.141592 / 4.0;
        double ry = (rand() / (double)RAND_MAX - 0.5) * 3.141592 / 5.0;
        double rz = (rand() / (double)RAND_MAX - 0.5) * 3.141592 / 4.0;

        double tx = (rand() / (double)RAND_MAX - 0.5) * 4;
        double ty = (rand() / (double)RAND_MAX - 0.5) * 4;
        double tz = rand() / (double)RAND_MAX * 1.0f;
        if (i > 3 * num_table_count / 4)
            tz = rand() / (double)RAND_MAX * 6.0 + 0.01;

        Eigen::Matrix3d m;
        m = Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX());

        for (auto& o : obj_pts)
        {
            Eigen::Vector3d transformed = m.transpose() * Eigen::Vector3d(o.x, o.y, o.z) +
                Eigen::Vector3d(tx, ty, tz);
            if (transformed.z() > 0)
            {
                auto ip = WorldToPlane(transformed, poly_inv, principal_point);
                if (ip.x() > 30 &&
                    ip.y() > 30 &&
                    ip.x() < img_size.x() - 30 &&
                    ip.y() < img_size.y() - 30)
                {
                    //cv::circle(tmp_img, cv::Point(ip.x(), ip.y()), 3, cv::Scalar(255), -1);
                    imgpts.push_back(cv::Point2f(ip.x(), ip.y()));
                    objpts.push_back(o);
                }
            }
        }
        if (imgpts.size() > 25)
        {
            num_total_pts += (int)imgpts.size();
            img_pts.push_back(std::move(imgpts));
            object_pts.push_back(std::move(objpts));
            //std::cout << tz << std::endl;
            //cv::imshow("tmp_img", tmp_img);
            //cv::waitKey(0);
            orig_ts.push_back(Eigen::Vector3d(tx, ty, tz));
            orig_R.push_back(m);
            orig_euler.push_back(Eigen::Vector3d(rx, ry, rz));
        }
    }

    ///calculate table



    cv::Mat K, xi, D;
    xi.setTo(0);
    std::vector<int> indices;
    std::vector<cv::Mat> rvecs, tvecs;
    //int flags = cv::omnidir::CALIB_FIX_SKEW | cv::omnidir::CALIB_FIX_XI;
    int flags = cv::omnidir::CALIB_FIX_SKEW;
    cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 2000, 0.00001);
    double rms;
    rms = cv::omnidir::calibrate(object_pts, img_pts,
        cv::Size(img_size.x(), img_size.y()),
        K, xi, D, rvecs, tvecs, flags, critia, indices);

    Eigen::Matrix3d K_eig;
    cv::cv2eigen(K, K_eig);
    K_out = K_eig.cast<float>();
    D_out[0] = (float)D.at<double>(0);
    D_out[1] = (float)D.at<double>(1);
    D_out[2] = (float)D.at<double>(2);
    D_out[3] = (float)D.at<double>(3);
    D_out[4] = 0.0;
    xi_out = (float)xi.at<double>(0);

    double sum_error_t = 0;
    double sum_error_rad = 0;


    for (size_t i = 0; i < indices.size(); ++i)
    {
        int j = indices[i];
        Eigen::Vector3d t(tvecs[i].at<double>(0), tvecs[i].at<double>(1), tvecs[i].at<double>(2));
        cv::Mat R;
        Eigen::Matrix3d R_eigen;
        cv::Rodrigues(rvecs[i], R);
        cv::cv2eigen(R, R_eigen);
        double tmp_err = (orig_ts[j] - t).norm();
        sum_error_t += tmp_err;
        if (tmp_err > 1)
        {
            cv::waitKey(0);
        }

        Eigen::Vector3d r = R_eigen.eulerAngles(0, 1, 2);
        Eigen::Vector3d r_orig = orig_R[j].transpose().eulerAngles(0, 1, 2);
        tmp_err = (r - r_orig).norm();
        sum_error_rad += tmp_err;
        if (tmp_err > 3.141592 / 180.0f)
        {
            cv::waitKey(0);
        }
    }

    std::cout << "Num tables total: " << indices.size();
    std::cout << "sum trans error in meter: " << sum_error_t << ", avg_err: " << sum_error_t / indices.size();
    std::cout << "sum rad error in degree: " << sum_error_rad * 180.0 / 3.141592 << ", avg_err: " << sum_error_rad * 180.0 / 3.141592 / indices.size();

    std::cout << "Converted parameters: \n";
    std::cout << "K:\n" << K;
    std::cout << "xi:\n" << xi;
    std::cout << "D:\n" << D;
    std::cout << "rms:" << rms << ", num_total_pts: " << num_total_pts;


    return rms;
}


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

    convertOcam2Mei(poly, poly_inv, principal_point, img_size, K_out, D_out, xi_out);
    return 0;
}
