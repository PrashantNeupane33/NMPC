#include <iostream>
#include <Eigen/Dense>
#include "Controller.hpp"
#include "Bezier.hpp"
#include "Observer.hpp"

using namespace Eigen;
using Eigen::placeholders::all;

void writeToCSV(const std::string& filename, const MatrixXd& matrix){
    const static IOFormat CSVFormat(Eigen::FullPrecision,Eigen::DontAlignCols, ", ", "\n");
	std::ofstream file(filename);
    if (file.is_open())
        file << matrix.format(CSVFormat);
    else
        std::cerr << "Error: Could not open " << filename << std::endl;
}

int main()
{
	unsigned int f = 30;
	unsigned int v = 26;
	double sampling = 0.05;

	MatrixXd C = MatrixXd::Identity(3, 3);

	VectorXd x0(3);
	x0 << 0.0, 0.0, 0.0;

	VectorXd u_min(3), u_max(3);
	u_min << -2.5, -2.5, -4.5;
	u_max <<  2.5,  2.5,  4.5;

	double Q0         = 25;
	double Qother     = 25;
	double predWeight = 200.0;

	auto horizons = std::make_tuple(v, f);
	auto weights  = std::make_tuple(Q0, Qother, predWeight);

	unsigned int timeSteps = 500 + f + 10;

	MatrixXd desiredTrajectory = getTrajectory(timeSteps, sampling);

	MPC mpc(C, horizons, weights, x0, desiredTrajectory, sampling, u_min, u_max);

	MatrixXd P0 = MatrixXd::Identity(3,3)*0.1;
    MatrixXd Q  = MatrixXd::Identity(3,3)*0.01;
    MatrixXd R  = MatrixXd::Identity(3,3)*0.5;

	EKF ekf(C, x0, P0, Q, R, sampling);

	VectorXd x_current = x0;
	int simSteps = timeSteps - f - 1;
	MatrixXd log_states(3, simSteps);
	MatrixXd log_inputs(3, simSteps);

	for(int i = 0; i < timeSteps - f - 1; i++){
		VectorXd u = mpc.computeControlInputs(x_current);
		VectorXd z = ekf.noisyMeasurement(x_current, u);

		ekf.predict(u);
		x_current = ekf.update(z);

		log_states.col(i) = x_current;
		log_inputs.col(i) = u;
	}

	// Data logs 
	writeToCSV("data/states.csv", log_states);
	writeToCSV("data/computedInputs.csv", log_inputs);
	writeToCSV("data/trajectory.csv", desiredTrajectory);

	std::cout << "Simulation completed!" << std::endl;
	return 0;
}
