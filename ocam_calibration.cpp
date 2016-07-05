#include "ocam_calibration.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <limits>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <cfloat>
#define MAX_LOG_LEVEL -100
#include <ceres/ceres.h>
//#include <unsupported/Eigen/src/Polynomials/PolynomialSolver.h>
using namespace std;
using namespace Eigen;

ceres::Solver s;
ceres::Problem pr;

struct CalibData
{
    vector<double> ss;
    vector<double> ss_inv;
    vector<Matrix3d> RRfin;
    double xc;
    double yc;
    MatrixXd Xt;
    MatrixXd Yt;
    vector<vector<cv::Point2f> > img_points;
    cv::Size img_size;
    int taylor_order_default;
};


vector<double> polyRootD2(double a, double b, double c)
{
    vector<double> ret;
    //Vector2d ret(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    double d = b * b - 4 * a * c;
    if (d > numeric_limits<double>::epsilon())
    {
        d = sqrt(d);
        double r = (-b + d) / (2 * a);
        //if (r > 0)
        ret.push_back(r);

        r = (-b - d) / (2 * a);
        //if (r > 0)
        ret.push_back(r);
    }
    else if (abs(d) < numeric_limits<double>::epsilon())
    {
        double r = -b / (2 * a);
        //if (r > 0)
        ret.push_back(r);
    }
    else
    {
        assert(false);
    }
    return ret;
}



bool vectorEqualsZero(const vector<double> & v)
{
    bool ret = true;

    for (auto e : v)
    {
        if (e != 0)
            return false;
    }

    return ret;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}



int plot_RR(const vector<Matrix3d> & RR,
    const Eigen::MatrixXd & Xt,
    const Eigen::MatrixXd & Yt,
    const Eigen::MatrixXd & Xpt,
    const Eigen::MatrixXd & Ypt)
{
    int index = -1;

    for (size_t i = 0; i < RR.size(); ++i)
    {
        Matrix3d RRdef = RR[i];
        double R11 = RRdef(0, 0);
        double R21 = RRdef(1, 0);
        double R31 = RRdef(2, 0);
        double R12 = RRdef(0, 1);
        double R22 = RRdef(1, 1);
        double R32 = RRdef(2, 1);
        double T1 = RRdef(0, 2);
        double T2 = RRdef(1, 2);

        Eigen::MatrixXd MA = (R21*Xt + R22*Yt).array() + T2;
        Eigen::MatrixXd MB = Ypt.cwiseProduct(R31 * Xt + R32 *Yt);
        Eigen::MatrixXd MC = (R11*Xt + R12*Yt).array() + T1;
        Eigen::MatrixXd MD = Xpt.cwiseProduct(R31*Xt + R32*Yt);
        Eigen::MatrixXd rho = (Xpt.cwiseProduct(Xpt) + Ypt.cwiseProduct(Ypt)).cwiseSqrt();
        Eigen::MatrixXd rho2 = rho.cwiseProduct(rho);
        //cout << MA << endl << endl;
        //cout << MB << endl << endl;
        //cout << MC << endl << endl;
        //cout << MD << endl << endl;
        //cout << rho << endl << endl;
        //cout << rho2 << endl << endl;

        Eigen::MatrixXd PP1(2 * Xt.rows(), 3);
        PP1.block(0, 0, Xt.rows(), 1) = MA;
        PP1.block(Xt.rows(), 0, Xt.rows(), 1) = MC;
        PP1.block(0, 1, Xt.rows(), 1) = MA.cwiseProduct(rho);
        PP1.block(Xt.rows(), 1, Xt.rows(), 1) = MC.cwiseProduct(rho);
        PP1.block(0, 2, Xt.rows(), 1) = MA.cwiseProduct(rho2);
        PP1.block(Xt.rows(), 2, Xt.rows(), 1) = MC.cwiseProduct(rho2);

        //PP1 = [MA, MA.*rho, MA.*rho2;
               //MC, MC.*rho, MC.*rho2];
        //cout << PP1 << endl << endl;

        Eigen::MatrixXd PP(2 * Xt.rows(), 4);
        PP.block(0, 0, PP1.rows(), 3) = PP1;
        PP.block(0, 3, Ypt.rows(), 1) = -Ypt;
        PP.block(Ypt.rows(), 3, Ypt.rows(), 1) = -Xpt;
        //cout << PP << endl << endl;
        //
        ////PP = [PP1, [-Ypt; -Xpt]];
        //
        Eigen::MatrixXd QQ (MB.rows() * 2, 1);
        QQ.block(0, 0, MB.rows(), 1) = MB;
        QQ.block(MB.rows(), 0, MB.rows(), 1) = MD;
        //cout << QQ << endl << endl;
        MatrixXd s = PP.jacobiSvd(ComputeThinU | ComputeThinV).solve(QQ);
        //cout << s << endl << endl;
        ////s = pinv(PP)*QQ;
        ////ss = s(1:3);
        ////if figure_number > 0
        ////    subplot(1, size(RR, 3), i); plot(0:620, polyval([ss(3) ss(2) ss(1)], [0:620])); grid; axis equal;
        ////end
        if (s(2) >= 0)
            index = (int)i;
       
    }
    return index;
}

std::vector<double> omni_find_parameters_fun(CalibData & cd)
{
    //fel van cserélve az x és az y
    double xc = cd.xc;
    double yc = cd.yc;

   
    int count = -1;

    Eigen::MatrixXd PP = Eigen::MatrixXd::Zero(2 * cd.Xt.rows() * cd.img_points.size(), cd.taylor_order_default + cd.img_points.size());
    Eigen::MatrixXd QQ = Eigen::MatrixXd::Zero(2 * cd.Xt.rows() * cd.img_points.size(), 1);

    for (size_t i = 0; i < cd.img_points.size(); ++i)
    {
        Eigen::MatrixXd Ypt(cd.img_points[i].size(), 1);
        Eigen::MatrixXd Xpt(cd.img_points[i].size(), 1);
        for (int j = 0; j < cd.img_points[i].size(); ++j)
        {
            Ypt(j, 0) = cd.img_points[i][j].x - yc;
            Xpt(j, 0) = cd.img_points[i][j].y - xc;
        }
        count = count + 1;

        Matrix3d RRdef = cd.RRfin[i];

        double R11 = RRdef(0, 0);
        double R21 = RRdef(1, 0);
        double R31 = RRdef(2, 0);
        double R12 = RRdef(0, 1);
        double R22 = RRdef(1, 1);
        double R32 = RRdef(2, 1);
        double T1 = RRdef(0, 2);
        double T2 = RRdef(1, 2);


        Eigen::MatrixXd MA = (R21*cd.Xt + R22*cd.Yt).array() + T2;
        Eigen::MatrixXd MB = Ypt.cwiseProduct(R31 * cd.Xt + R32 *cd.Yt);
        Eigen::MatrixXd MC = (R11*cd.Xt + R12*cd.Yt).array() + T1;
        Eigen::MatrixXd MD = Xpt.cwiseProduct(R31*cd.Xt + R32*cd.Yt);
        vector<Eigen::MatrixXd> rho(cd.taylor_order_default);
            
        //rho = [];
        //for j = 2:taylor_order
        //    rho(:, : , j) = (sqrt(Xpt. ^ 2 + Ypt. ^ 2)).^j;
        //end
        
        MatrixXd tmp = (Xpt.cwiseProduct(Xpt) + Ypt.cwiseProduct(Ypt)).cwiseSqrt();
        rho[0] = tmp;
        for (size_t j = 1; j < cd.taylor_order_default; ++j)
        {
            rho[j] = tmp.cwiseProduct(rho[j -1]);
        }
        rho[0] = MatrixXd::Zero(tmp.rows(), tmp.cols());

      
        MatrixXd PP1(cd.Xt.rows() * 2, cd.taylor_order_default);
        PP1.block(0,0, cd.Xt.rows(), 1)= MA;
        PP1.block(cd.Xt.rows(), 0, cd.Xt.rows(), 1) = MC;
        for (int j = 1; j < cd.taylor_order_default; ++j)
        {
            PP1.block(0, j, cd.Xt.rows(), 1) = MA.cwiseProduct(rho[j]);
            PP1.block(cd.Xt.rows(), j, cd.Xt.rows(), 1) = MC.cwiseProduct(rho[j]);
        }

        
        PP.block(PP1.rows() * i, 0, PP1.rows(), PP1.cols()) = PP1;
        PP.block(PP1.rows() * i, cd.taylor_order_default + i, Ypt.rows(), 1) = -Ypt;
        PP.block(PP1.rows() * i + Ypt.rows(), cd.taylor_order_default + i, Ypt.rows(), 1) = -Xpt;
        //cout << PP;
            //PP = [PP   zeros(size(PP, 1), 1);
        //PP1, zeros(size(PP1, 1), count - 1)[-Ypt; -Xpt]];
         
        QQ.block(PP1.rows() * i, 0, MB.rows(), 1) = MB;
        QQ.block(PP1.rows() * i + MB.rows(), 0, MB.rows(), 1) = MD;
        //cout << QQ;
    }
    //cout << PP << endl << endl;
    //cout << QQ << endl << endl;

    MatrixXd s = PP.jacobiSvd(ComputeThinU | ComputeThinV).solve(QQ);
    //cout << s << endl << endl;

    vector<double> ret(cd.taylor_order_default + 1);
    ret[0] = s(0);
    ret[1] = 0;
    for (int i = 2; i < cd.taylor_order_default + 1; ++i)
        ret[i] = s(i - 1);

    for (size_t i = 0; i < cd.img_points.size(); ++i)
        cd.RRfin[i](2, 2) = s(cd.taylor_order_default + i);
    cd.ss = ret;
    return ret;
}

template <typename T>
T polyval(const vector<T> & ss,
    T x)
{
    T ret = T(0);
    T m = T(1);
    for (auto & c : ss)
    {
        ret += c * m;
        m *= x;
    }
    return ret;
}

double polyval2(const vector<double> & ss, 
    double theta, double radius)
{
    double m = tan(theta);
    std::vector<double> poly_coeff_tmp = ss;
    poly_coeff_tmp[1] = poly_coeff_tmp[1] - m;
    vector<double> p_deriv_tmp(poly_coeff_tmp.size() - 1);
    for (size_t i = 0; i < p_deriv_tmp.size(); ++i)
    {
        p_deriv_tmp[i] = (i + 1) * poly_coeff_tmp[i + 1];
    }
    int max_iter_count = 100;
    double x0 = 0;
    double tolerance = 10e-14;
    double epsilon = 10e-14;
    bool haveWeFoundSolution = false;
    for (int i = 0; i < max_iter_count; ++i)
    {
        double y = polyval(poly_coeff_tmp, x0);
        double yprime = polyval(p_deriv_tmp, x0);

        if (abs(yprime) < epsilon) // Don't want to divide by too small of a number
            // denominator is too small
            break;                                        //Leave the loop
            //end
        double x1 = x0 - y / yprime;

        if (abs(x1 - x0) <= tolerance * abs(x1))
        {
            haveWeFoundSolution = true;
            break;
        }
        x0 = x1;
    }
    if (haveWeFoundSolution)
    {
        double tmp = polyval(poly_coeff_tmp, x0);
        return x0;
    }
    else
        return numeric_limits<double>::quiet_NaN();
}

vector<double> invertOcamPoly(const std::vector<double> & p)
{
    vector<double> theta;
    vector<double> r;

    double step = 0.01;
    double width = 1280;
    double height = 1080;
    double radius = sqrt(width * width / 4 + height * height / 4);
    for (double x = -M_PI_2; x < 1.2; x += step)
    {
        double y = polyval2(p, x, radius);
        if (y != numeric_limits<double>::quiet_NaN() && y > 0)
        {
            theta.push_back(x);
            r.push_back(y);
        }
    }

    int poly_degree = 14;

    MatrixXd A(theta.size(), poly_degree + 1);
    MatrixXd b(theta.size(), 1);

    for (size_t i = 0; i < theta.size(); ++i)
    {
        A(i, 0) = 1;
        b(i) = r[i];
        for (size_t j = 1; j < poly_degree + 1; ++j)
        {
            A(i, j) = A(i, j - 1) * theta[i];
        }
    }
    //cout << A << endl << endl;
    MatrixXd x = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

    vector<double> ret(poly_degree + 1);
    for (int i = 0; i < poly_degree + 1; ++i)
        ret[i] = x(i);
    return ret;
}

template <typename T>
Matrix<T,2,1> reprojectPoint(const std::vector<T> & ss_inv, const Matrix<T, 3, 1> & xx, cv::Size img_size)
{
    T d_s = ceres::sqrt(xx(0) * xx(0) + xx(1) * xx(1));
    T m = ceres::atan2( xx(2) , d_s);
    T rho = polyval(ss_inv, m);
    return Matrix<T, 2, 1>(xx(0) / d_s * rho, xx(1) / d_s * rho);
}



double reprojectError(CalibData & cd)
{
    double rms2 = 0;
    int count = 0;
    for (auto k = 0; k < cd.img_points.size(); ++k)
    {
        //reprojecting points
        for (int i = 0; i < cd.Xt.rows(); ++i)
        {
            Vector3d w = cd.RRfin[k] * Vector3d(cd.Xt(i), cd.Yt(i), 1);
            Vector2d reprojected = reprojectPoint(cd.ss_inv, w, cd.img_size);

            Vector2d orig(cd.img_points[k][i].y - cd.xc, cd.img_points[k][i].x - cd.yc);
            //cout << w << " in pixels " << reprojected << endl;
            rms2 += (orig - reprojected).norm();
            ++count;

        }
    }

    return rms2 / count;
}

double reprojectErrorSquaredSum(CalibData & cd)
{
	double rms2 = 0;
	for (auto k = 0; k < cd.img_points.size(); ++k)
	{
		//reprojecting points
		for (int i = 0; i < cd.Xt.rows(); ++i)
		{
			Vector3d w = cd.RRfin[k] * Vector3d(cd.Xt(i), cd.Yt(i), 1);
			Vector2d reprojected = reprojectPoint(cd.ss_inv, w, cd.img_size);

			Vector2d orig(cd.img_points[k][i].y - cd.xc, cd.img_points[k][i].x - cd.yc);
			//cout << w << " in pixels " << reprojected << endl;
			rms2 += (orig - reprojected).squaredNorm();
			;

		}
	}

	return rms2;
}

bool isMatricesZeros(const vector<Matrix3d> & ms)
{
    for (auto & m : ms)
    {
        if (m.norm() < numeric_limits<double>::epsilon())
            return true;
    }
    return false;
}



double calibrateCameraOcam(CalibData & calib_data)
{
    for (int kk = 0; kk < calib_data.img_points.size(); ++kk)
    {
        Eigen::MatrixXd Ypt(calib_data.img_points[kk].size(), 1);
        Eigen::MatrixXd Xpt(calib_data.img_points[kk].size(), 1);
        for (int j = 0; j < calib_data.img_points[kk].size(); ++j)
        {
                Ypt(j, 0) = calib_data.img_points[kk][j].x - calib_data.yc;
                Xpt(j, 0) = calib_data.img_points[kk][j].y - calib_data.xc;
        }

        //cout << Xpt << endl;
        //cout << Ypt << endl;

        Eigen::MatrixXd A(calib_data.img_points[kk].size(), 6);

        A.col(0) = calib_data.Xt.cwiseProduct(Ypt);
        A.col(1) = calib_data.Yt.cwiseProduct(Ypt);
        A.col(2) = -calib_data.Xt.cwiseProduct(Xpt);
        A.col(3) = -calib_data.Yt.cwiseProduct(Xpt);
        A.col(4) = Ypt;
        A.col(5) = -Xpt;

        //cout << A << endl;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);

        double R11 = svd.matrixV()(0, 5);
        double R12 = svd.matrixV()(1, 5);
        double R21 = svd.matrixV()(2, 5);
        double R22 = svd.matrixV()(3, 5);
        double T1 = svd.matrixV()(4, 5);
        double T2 = svd.matrixV()(5, 5);

        double AA = ((R11*R12) + (R21*R22)) *  ((R11*R12) + (R21*R22));
        double BB = R11 * R11 + R21 * R21;
        double CC = R12 *R12 + R22 * R22;
        vector<double> R32_2 = polyRootD2(1, CC - BB, -AA);


        vector<double> R31;
        vector<double> R32;
        vector<double> sg = { 1.0, -1.0 };
        for (auto r : R32_2)
        {
            if (r > 0)
            {
                for (auto c : sg)
                {
                    double sqrtR32_2 = c * sqrt(r);
                    R32.push_back(sqrtR32_2);
                    if (vectorEqualsZero(R32_2))
                    {
                        R31.push_back(sqrt(CC - BB));
                        R31.push_back(-sqrt(CC - BB));
                        R32.push_back(sqrtR32_2);
                    }
                    else
                    {
                        R31.push_back((R11*R12 + R21*R22) / -sqrtR32_2);
                    }
                }
            }
        }

        std::vector<Eigen::Matrix3d> RR(R32.size() * 2);

        int count = -1;
        for (size_t i1 = 0; i1 < R32.size(); ++i1)
        {
            for (size_t i2 = 0; i2 < 2; ++i2)
            {
                count = count + 1;
                double Lb = 1 / sqrt(R11 * R11 + R21 * R21 + R31[i1] * R31[i1]);
                RR[count] << R11, R12, T1,
                    R21, R22, T2,
                    R31[i1], R32[i1], 0.0;
                RR[count] *= sg[i2] * Lb;
            }
        }



        double minRR = numeric_limits<double>::infinity();
        int minRR_ind = -1;
        for (size_t min_count = 0; min_count < RR.size(); ++min_count)
        {
            if ((Vector2d(RR[min_count](0, 2), RR[min_count](1, 2)) - Vector2d(Xpt(0), Ypt(0))).norm() < minRR)
            {
                minRR = (Vector2d(RR[min_count](0, 2), RR[min_count](1, 2)) - Vector2d(Xpt(0), Ypt(0))).norm();
                minRR_ind = (int)min_count;
            }
        }

        std::vector<Eigen::Matrix3d> RR1;
        if (minRR_ind != -1)
        {
            for (size_t count = 0; count < RR.size(); ++count)
            {
                if (sgn(RR[count](0, 2)) == sgn(RR[minRR_ind](0, 2)) &&
                    sgn(RR[count](1, 2)) == sgn(RR[minRR_ind](1, 2)))
                {
                    RR1.push_back(RR[count]);
                    //cout << RR[count] << endl;
                }
            }
        }

        if (RR1.empty())
        {
            //RRfin = 0;
            //ss = 0;
            return numeric_limits<double>::infinity();
        }

        //R32_2 = roots([1, CC - BB, -AA]);
        //R32_2 = R32_2(find(R32_2 >= 0));
        int nm = plot_RR(RR1, calib_data.Xt, calib_data.Yt, Xpt, Ypt);
        Matrix3d RRdef = RR1[nm];
        calib_data.RRfin[kk] = RRdef;
    }
    int taylor_order_default = 4;
    std::vector<double> ss = omni_find_parameters_fun(calib_data);


    calib_data.ss_inv = invertOcamPoly(ss);


    double avg_reprojection_error = reprojectError(calib_data);


    return avg_reprojection_error;
}


void findCenter(CalibData & cd)
{
    double pxc = cd.xc;
    double pyc = cd.yc;
    double width = cd.img_size.width;
    double height = cd.img_size.height;
    double regwidth = (width / 2);
    double regheight = (height / 2);
    double yceil = 5;
    double xceil = 5;

    double xregstart = pxc - (regheight / 2);
    double xregstop = pxc + (regheight / 2);
    double yregstart = pyc - (regwidth / 2);
    double yregstop = pyc + (regwidth / 2);
    //cout << "Iteration 1." << endl;
    for (int glc = 0; glc < 9; ++glc)
    {
        double s = ((yregstop - yregstart) / yceil);
        int c = int((yregstop - yregstart) / s + 1);

        MatrixXd yreg(c, c);
        for (int i = 0; i < c; ++i)
        {
            for (int j = 0; j < c; ++j)
            {
                yreg(i, j) = yregstart + j * s;
            }

        }
        //cout << yreg << endl;

        s = ((xregstop - xregstart) / xceil);
        //c = (xregstop - xregstart) / s + 1;

        MatrixXd xreg(c, c);
        for (int i = 0; i < c; ++i)
        {
            for (int j = 0; j < c; ++j)
            {
                xreg(i, j) = xregstart + i * s;
            }

        }
        //cout << xreg << endl;


        int ic_proc = (int)xreg.rows();
        int jc_proc = (int)xreg.cols();
        //MatrixXd MSEA = numeric_limits<double>::infinity() * MatrixXd::Ones(xreg.rows(), xreg.cols());
        int min_idx1, min_idx2;
        double min_MSEA = numeric_limits<double>::max();
        for (int ic = 0; ic < ic_proc; ++ic)
        {
            for (int jc = 0; jc < jc_proc; ++jc)
            {
                cd.xc = xreg(ic, jc);
                cd.yc = yreg(ic, jc);

                calibrateCameraOcam(cd);
                if (isMatricesZeros(cd.RRfin))
                {
                    // MSEA(ic, jc) = numeric_limits<double>::infinity();
                    continue;
                }
                double MSE = reprojectError(cd);

                if (!isnan(MSE))
                {
                    if (MSE < min_MSEA)
                    {
                        min_MSEA = MSE;
                        min_idx1 = ic;
                        min_idx2 = jc;
                    }
                    //MSEA(ic, jc) = MSE;
                }
            }
            //    %obrand_end
        }

        //    %    drawnow;
        //indMSE = find(min(MSEA(:)) == MSEA);
        cd.xc = xreg(min_idx1, min_idx2);
        cd.yc = yreg(min_idx1, min_idx2);
        double dx_reg = abs((xregstop - xregstart) / xceil);
        double dy_reg = abs((yregstop - yregstart) / yceil);
        xregstart = cd.xc - dx_reg;
        xregstop = cd.xc + dx_reg;
        yregstart = cd.yc - dy_reg;
        yregstop = cd.yc + dy_reg;
        //cout << glc << endl;
    }


}
#include <ceres/rotation.h>

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double o_x, double o_y, const CalibData & cd, double wx, double wy)
        : observed_x(o_x), observed_y(o_y), calib_data(cd), world_x(wx), world_y(wy) {}

   
    template <typename T>
    bool operator()(const T* const camera,
        T* residuals
		) const 
    {
        // camera[0,1,2] are the angle-axis rotation.
        //Vector3d p;
        Matrix<T, 3, 1> p;
        Matrix3d R;
        ceres::AngleAxisToRotationMatrix<T>(camera, R.data());
        //cout << R * R.transpose() << endl;
        R(0, 2) = camera[3];
        R(1, 2) = camera[4];
        R(2, 2) = camera[5];
        p(0) = world_x;
        p(1) = world_y;
        p(2) = 1;
        p = R * p;

      
        Matrix<T,2,1> ret = reprojectPoint<T>(calib_data.ss_inv, p, calib_data.img_size);
        //cout << T(observed_y - calib_data.xc - ret.x()) << endl;
        //cout << T(observed_x - calib_data.yc - ret.y()) << endl;
        // The error is the difference between the predicted and observed position.
        residuals[0] = T(observed_y - calib_data.xc - ret.x());
        residuals[1] = T(observed_x - calib_data.yc - ret.y());
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
   // static ceres::CostFunction* Create(const double observed_x,
   //     const double observed_y, const CalibData & cd) {
   //     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
   //         new SnavelyReprojectionError(observed_x, observed_y,cd)));
   // }

    static ceres::CostFunction* CreateNumericDiff(const double observed_x,
        const double observed_y, const CalibData & cd, double wx, double wy) {
        return (new ceres::NumericDiffCostFunction<SnavelyReprojectionError,ceres::CENTRAL, 2, 6>(
            new SnavelyReprojectionError(observed_x, observed_y, cd, wx, wy)));
    }

    double observed_x;
    double observed_y;
    double world_x;
    double world_y;

    const CalibData & calib_data;
};


struct IntinsicsReprojectionError {
	IntinsicsReprojectionError(const CalibData &  cd) : calib_data(cd) {}


	template <typename T>
	bool operator()(const T* const camera,
		T* residuals
		) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		//Vector3d p;
		calib_data.xc = camera[0];
		calib_data.yc = camera[1];
		for (size_t i = 0; i < calib_data.ss.size(); ++i)
			calib_data.ss[i] = camera[2 + i];

		calib_data.ss_inv = invertOcamPoly(calib_data.ss);

		//cout << T(observed_y - calib_data.xc - ret.x()) << endl;
		//cout << T(observed_x - calib_data.yc - ret.y()) << endl;
		// The error is the difference between the predicted and observed position.
		residuals[0] = T(reprojectErrorSquaredSum(calib_data));
		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	// static ceres::CostFunction* Create(const double observed_x,
	//     const double observed_y, const CalibData & cd) {
	//     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
	//         new SnavelyReprojectionError(observed_x, observed_y,cd)));
	// }

	static ceres::CostFunction* CreateNumericDiff(const CalibData & cd) {
		return (new ceres::NumericDiffCostFunction<IntinsicsReprojectionError, ceres::CENTRAL, 1, 7>(
			new IntinsicsReprojectionError(cd)));
	}

	mutable CalibData calib_data;
};

double refineParameters(CalibData & cd, int iter_count)
{
	double MSE_tol = 1e-8;
	double MSE_old = 0;
	double MSE_new = numeric_limits<double>::max();
	for (int iter = 0; iter < iter_count && abs(MSE_new - MSE_old) > MSE_tol; ++iter)
	{
		//vector<Vector2d> center(cd.img_points.size(), Vector2d(cd.xc, cd.yc));
		//optimizing chessboard positions
		for (int kk = 0; kk < cd.img_points.size(); ++kk)
		{

			Matrix3d R = cd.RRfin[kk];
			R.col(2) = R.col(0).cross(R.col(1));
			Vector3d t = cd.RRfin[kk].col(2);
			Vector3d r;
			ceres::RotationMatrixToAngleAxis(R.data(), r.data());


			//cout << tt - R << endl;
			//cout <<R * R.transpose() << endl;

			Matrix<double, 6, 1> refined_RT;
			refined_RT.block(0, 0, 3, 1) = r;
			refined_RT.block(3, 0, 3, 1) = t;


			ceres::Problem problem;
			for (int j = 0; j < cd.img_points[kk].size(); ++j)
			{
				Vector3d p(cd.Xt(j), cd.Yt(j), 1.0);

				ceres::CostFunction* cost_function =
					SnavelyReprojectionError::CreateNumericDiff(
						cd.img_points[kk][j].x,
						cd.img_points[kk][j].y, cd, p.x(), p.y());
				problem.AddResidualBlock(cost_function,
					NULL /* squared loss */,
					refined_RT.data());

				//pr.AddResidualBlock();
			}


			ceres::Solver::Options options;
			options.linear_solver_type = ceres::DENSE_SCHUR;
			options.minimizer_progress_to_stdout = false;
			options.logging_type = ceres::SILENT;
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			//std::cout << summary.FullReport() << "\n";
			//std::cout << center << "\n";

			r = refined_RT.block(0, 0, 3, 1);
			//cout << t - refined_RT.block(3, 0, 3, 1) << endl;
			t = refined_RT.block(3, 0, 3, 1);


			ceres::AngleAxisToRotationMatrix(r.data(), cd.RRfin[kk].data());
			//cout << cd.RRfin[kk] * cd.RRfin[kk].transpose() << endl;
			cd.RRfin[kk].col(2) = t;

		}

		//optimizing the intrinsics values
		ceres::CostFunction* cost_function =
			IntinsicsReprojectionError::CreateNumericDiff(cd);
		ceres::Problem problem;

		Matrix<double, 7, 1> refined_params;
		refined_params[0] = cd.xc;
		refined_params[1] = cd.yc;
		for (size_t i = 0; i < cd.ss.size(); ++i)
			refined_params[2 + i] = cd.ss[i];

		problem.AddResidualBlock(cost_function,
			NULL /* squared loss */,
			refined_params.data());
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.logging_type = ceres::SILENT;
		options.minimizer_progress_to_stdout = false;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//ceres::Solve(options, &problem, nullptr);
		//std::cout << summary.FullReport() << "\n";


		cd.xc = refined_params[0];
		cd.yc = refined_params[1];
		for (size_t i = 0; i < cd.ss.size(); ++i)
			cd.ss[i] = refined_params[2 + i];

		cd.ss_inv = invertOcamPoly(cd.ss);
		MSE_old = MSE_new;
		MSE_new = reprojectError(cd);;
	}
    return reprojectError(cd);
}

double calibrateCameraOcam2(const vector<vector<cv::Point3f> > & objectPoints,
    const vector<vector<cv::Point2f> > & imagePoints, 
    const cv::Size & imageSize, OcamCalibRes & res)
{
    
	google::InitGoogleLogging("ocam");
    CalibData cd;
    cd.img_size = imageSize;
    cd.taylor_order_default = 4;
    cd.img_points = imagePoints;


    //fel van cserélve az x és az y
    cd.xc = imageSize.height / 2.0;
    cd.yc = imageSize.width / 2.0;

    cd.Yt = Eigen::MatrixXd(objectPoints[0].size(), 1);
    cd.Xt = Eigen::MatrixXd(objectPoints[0].size(), 1);
    for (int j = 0; j < objectPoints[0].size(); ++j)
    {
        cd.Yt(j, 0) = objectPoints[0][j].x;
        cd.Xt(j, 0) = objectPoints[0][j].y; 
    }

   cd.RRfin.resize(objectPoints.size());

   double avg_reprojection_error = calibrateCameraOcam(cd);
   //cout << "Ocam Average reprojection error befor find center: " << avg_reprojection_error << endl;
   findCenter(cd);
   avg_reprojection_error = calibrateCameraOcam(cd);
   //cout << "Ocam Average reprojection error : " << avg_reprojection_error << endl;
   //cout << "img center: " << cd.xc << "  : " << cd.yc << endl;

   avg_reprojection_error = refineParameters(cd, 100);
   //cout << "Ocam Average reprojection error after refinement : " << avg_reprojection_error << endl;

   res.center_x = cd.xc;
   res.center_y = cd.yc;

   res.ss = cd.ss;
   res.ss_inv = cd.ss_inv;

   for (auto v : cd.RRfin)
   {
	   Matrix3d R = v;
	   R.col(2) = R.col(0).cross(R.col(1));

	   res.R.push_back({
		   R(0,0), R(0, 1), R(0, 2),
		   R(1,0), R(1, 1), R(1, 2),
		   R(2,0), R(2, 1), R(2, 2)
	   });

	   Vector3d t = v.col(2);
	   res.T.push_back({t(0), t(1), t(2)});
   }

   return avg_reprojection_error;
}
