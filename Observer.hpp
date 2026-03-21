#ifndef EXTENDED_KALMAN_FILTER_H
#define EXTENDED_KALMAN_FILTER_H

#include <Eigen/Dense>
#include <random>
using namespace Eigen;

class EKF {

    private:
        unsigned int n; // state dimension
        unsigned int m; // input dimension
        unsigned int r; // measurement dimension

        double sampling; // dt

        MatrixXd C; // measurement matrix

        VectorXd x_est; // current state estimate
        MatrixXd P; // state covariance
        MatrixXd Q; // process noise covariance
        MatrixXd R; // measurement noise covariance
	
		std::default_random_engine gen;
		std::normal_distribution<double> xy_noise;
		std::normal_distribution<double> th_noise;

        // Compute Jacobian of dynamics w.r.t. state  (F = df/dx)
        MatrixXd jacobianF(const VectorXd& x, const VectorXd& u) const;

        // Compute Jacobian of measurement model w.r.t. state  (H = dh/dx)
        MatrixXd jacobianH() const;

    public:
        EKF(MatrixXd _C,
            VectorXd _x0,
            MatrixXd _P0,
            MatrixXd _Q,
            MatrixXd _R,
            double _sampling);

        // Propagate state through nonlinear dynamics
        VectorXd dynamics(const VectorXd& x, const VectorXd& u) const;
        void predict(const VectorXd& u); // Prediction step
        VectorXd update(const VectorXd& z);// Update step
		VectorXd noisyMeasurement(const VectorXd& x, const VectorXd& u);
};
#endif
