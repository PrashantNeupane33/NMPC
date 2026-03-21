#ifndef MODELPREDICTIVECONTROLLER_H
#define MODELPREDICTIVECONTROLLER_H

#include<string>
#include<tuple>
#include<Eigen/Dense>
#include <casadi/casadi.hpp>
using namespace Eigen;

class MPC{

    private:
        unsigned int k; // current time step
        unsigned int m,n,r; // (m,n,r) -> (input, state, output) dimension
		double sampling; // dt

	    MatrixXd C; // system matrices
		std::tuple<double, double, double> weights; //(Q0,Qother,predWeight)
        MatrixXd desiredInput; // total desired trajectory
        unsigned int f,v; // f-> prediction horizon, v -> control horizon

        MatrixXd inputs;
        MatrixXd outputs;

		// Lifted state matrix. O -> observability matrix, M -> Toeplitz matrix
        MatrixXd O;
        MatrixXd M;

        // control gain matrix
        MatrixXd gainMatrix;
		
		VectorXd u_prev;
		VectorXd u_min, u_max; // actuator limits

		casadi::Function casadi_solver;
		casadi::Function casadi_f;
		bool solver_initialized = false;

		void initCasADiSolver();
		std::tuple<MatrixXd, MatrixXd> linearizeModel(const VectorXd& x_bar,const VectorXd& u_bar) const;
		void setObservabilityMatrix(const MatrixXd& A_k);
		void setToeplitzMatrix(const MatrixXd& A_k, const MatrixXd& B_k);
		void setGainMatrix(std::tuple<double, double, double> weights);
		std::tuple<MatrixXd,MatrixXd> getWeightMatrices(std::tuple<double, double, double> weights);
		void writeToCSV(const std::string& filename, const MatrixXd& matrix) const;

    public:
		MPC(MatrixXd _C,
			std::tuple<unsigned int, unsigned int> horizons,
			std::tuple<double, double, double> weights,
			VectorXd _initialState,
			MatrixXd _desiredTrajectory,
			double _sampling,
			VectorXd _u_min,
			VectorXd _u_max);

        VectorXd computeControlInputs(VectorXd x_k);
};
#endif
