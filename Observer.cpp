#include <Eigen/Dense>
#include "Observer.hpp"
#include <cmath>
#include <random>

using namespace Eigen;

EKF::EKF(MatrixXd _C,
         VectorXd _x0,
         MatrixXd _P0,
         MatrixXd _Q,
         MatrixXd _R,
         double   _sampling) :
    C(_C),
    x_est(_x0),
    P(_P0),
    Q(_Q),
    R(_R),
    sampling(_sampling)
{
    n = (unsigned int)_x0.size();   // state dimension  (3)
    m = 3;                           // input dimension  (3)
    r = (unsigned int)_C.rows();    // measurement dimension
	
	xy_noise = std::normal_distribution<double>(0.0, std::sqrt(_R(0,0)));
    th_noise = std::normal_distribution<double>(0.0, std::sqrt(_R(2,2)));
}

VectorXd EKF::dynamics(const VectorXd& x, const VectorXd& u) const
{
    double theta = x(2);
    VectorXd x_dot(n);
    x_dot(0) = u(0)*cos(theta) - u(1)*sin(theta);
    x_dot(1) = u(0)*sin(theta) + u(1)*cos(theta);
    x_dot(2) = u(2);
    return x + sampling * x_dot;
}

MatrixXd EKF::jacobianF(const VectorXd& x, const VectorXd& u) const
{
    double theta  = x(2);
    double vx     = u(0);
    double vy     = u(1);

    MatrixXd F = MatrixXd::Identity(n, n);
    F(0, 2)    = sampling * (-vx*sin(theta) - vy*cos(theta));
    F(1, 2)    = sampling * ( vx*cos(theta) - vy*sin(theta));

    return F;
}

MatrixXd EKF::jacobianH() const
{
    return C;
}

VectorXd EKF::noisyMeasurement(const VectorXd& x, const VectorXd& u)
{
    VectorXd z = dynamics(x, u);
    z(0) += xy_noise(gen);
    z(1) += xy_noise(gen);
    z(2) += th_noise(gen);
    return z;
}

void EKF::predict(const VectorXd& u)
{
	// Covariance
	MatrixXd F = jacobianF(x_est, u);
    P = F * P * F.transpose() + Q;

    // State
    x_est = dynamics(x_est, u);

    // Angle wrap
    while (x_est(2) >  M_PI) x_est(2) -= 2*M_PI;
    while (x_est(2) < -M_PI) x_est(2) += 2*M_PI;
}

VectorXd EKF::update(const VectorXd& z)
{
    MatrixXd H = jacobianH();

    // Innovation
    VectorXd y = z - H * x_est;

    // Angle wrap
    while (y(2) >  M_PI) y(2) -= 2*M_PI;
    while (y(2) < -M_PI) y(2) += 2*M_PI;

    // Innovation covariance
    MatrixXd S = H * P * H.transpose() + R;

    // Kalman gain:
    MatrixXd K = P * H.transpose() * S.inverse();

    // State update
    x_est = x_est + K * y;

    // Angle wrap
    while (x_est(2) >  M_PI) x_est(2) -= 2*M_PI;
    while (x_est(2) < -M_PI) x_est(2) += 2*M_PI;

    // Covariance
    MatrixXd IKH = MatrixXd::Identity(n, n) - K * H;
    P = IKH * P * IKH.transpose() + K * R * K.transpose();

	return x_est;
}
