#include<iostream>
#include<string>
#include<fstream>
#include<tuple>
#include<cmath>
#include<Eigen/Dense>
#include <casadi/casadi.hpp>
#include "Controller.hpp"

using namespace Eigen;
using namespace std;
using Eigen::placeholders::all;

MPC::MPC(MatrixXd _C,
         std::tuple<unsigned int, unsigned int> horizons,
         std::tuple<double, double, double> weights,
         VectorXd _initialState,
         MatrixXd _desiredTrajectory,
         double _sampling,
         VectorXd _u_min,
         VectorXd _u_max):
    C(_C),
    f(get<1>(horizons)), v(get<0>(horizons)),
    x0(_initialState),
    desiredInput(_desiredTrajectory),
    sampling(_sampling),
    u_min(_u_min),
    u_max(_u_max),
    k(0)
{
    n = 3;
    m = _u_min.size();
    r = C.rows();

    unsigned int maxSimulationSamples = desiredInput.rows() - f;

    states.resize(n, maxSimulationSamples);
    states.setZero();
    states.col(0) = x0;

    inputs.resize(m, maxSimulationSamples - 1);
    inputs.setZero();

    outputs.resize(r, maxSimulationSamples - 1);
    outputs.setZero();

    u_prev = VectorXd::Zero(m);

    auto wm = getWeightMatrices(weights);
    W3 = get<0>(wm);
    W4 = get<1>(wm);

	initCasADiSolver();
}

void MPC::initCasADiSolver()
{
	casadi::MX x_sym = casadi::MX::sym("x", (int)n);
	casadi::MX u_sym = casadi::MX::sym("u", (int)m);

	// System Dynamics
	casadi::MX theta = x_sym(2);

	std::vector<casadi::MX> xdot_vec = {
		u_sym(0)*casadi::MX::cos(theta) - u_sym(1)*casadi::MX::sin(theta),
		u_sym(0)*casadi::MX::sin(theta) + u_sym(1)*casadi::MX::cos(theta),
		u_sym(2)
	};
	casadi::MX x_dot  = casadi::MX::vertcat(xdot_vec);
	casadi::MX x_next = x_sym + sampling * x_dot;
	casadi_f = casadi::Function("f", {x_sym, u_sym}, {x_next});

	// Decision variable and parameter vector
	casadi::MX W       = casadi::MX::sym("W", (int)((n+m)*f + n));
	casadi::MX P_param = casadi::MX::sym("P", (int)(n + (f+1)*n));

	// Weight Matrices
	casadi::MX Q_cas = casadi::MX::zeros((int)n, (int)n);
	Q_cas(0,0) = 500.0;
	Q_cas(1,1) = 500.0;
	Q_cas(2,2) = 1.0;

	casadi::MX R_cas = casadi::MX::zeros((int)m, (int)m);
	R_cas(0,0) = 0.1;
	R_cas(1,1) = 0.1;
	R_cas(2,2) = 0.05;

	casadi::MX P_cas = casadi::MX::zeros((int)n, (int)n);
	P_cas(0,0) = 1000.0;
	P_cas(1,1) = 1000.0;
	P_cas(2,2) = 2.0;

	// Initial condition constraint: x_0 = x_current
	casadi::MX x0_p = P_param(casadi::Slice(0, (int)n));
	casadi::MX X_k  = W(casadi::Slice(0, (int)n));

	std::vector<casadi::MX> g;
	std::vector<double> lbg, ubg;

	g.push_back(X_k - x0_p);
	for(int i = 0; i < (int)n; i++) {
		lbg.push_back(0.0);
		ubg.push_back(0.0);
	}

	// Horizon Loop
	casadi::MX J = 0;

	for(int k = 0; k < (int)f; k++)
	{
		int x_idx = k * ((int)n + (int)m);
		int u_idx = x_idx + (int)n;

		casadi::MX Xk  = W(casadi::Slice(x_idx,              x_idx + (int)n));
		casadi::MX Uk  = W(casadi::Slice(u_idx,              u_idx + (int)m));
		casadi::MX Xk1 = W(casadi::Slice(x_idx + (int)(n+m), x_idx + (int)(n+m) + (int)n));

		casadi::MX x_ref = P_param(casadi::Slice((int)n + k*(int)n, (int)n + (k+1)*(int)n));

		casadi::MX err = Xk - x_ref;

		std::vector<casadi::MX> ew_vec = {
			err(0),
			err(1),
			casadi::MX::atan2(casadi::MX::sin(err(2)), casadi::MX::cos(err(2)))
		};
		casadi::MX err_wrapped = casadi::MX::vertcat(ew_vec);

		J += casadi::MX::mtimes({err_wrapped.T(), Q_cas, err_wrapped})
		   + casadi::MX::mtimes({Uk.T(), R_cas, Uk});

		std::vector<casadi::MX> f_args = {Xk, Uk};
		casadi::MX x_pred = casadi_f(f_args)[0];

		g.push_back(Xk1 - x_pred);
		for(int i = 0; i < (int)n; i++) {
			lbg.push_back(0.0);
			ubg.push_back(0.0);
		}
	}

	// Terminal Cost
	casadi::MX x_ref_N = P_param(casadi::Slice((int)n + (int)f*(int)n, (int)n + ((int)f+1)*(int)n));
	casadi::MX X_N     = W(casadi::Slice((int)f*((int)n+(int)m), (int)f*((int)n+(int)m) + (int)n));
	casadi::MX err_N   = X_N - x_ref_N;

	std::vector<casadi::MX> en_vec = {
		err_N(0),
		err_N(1),
		casadi::MX::atan2(casadi::MX::sin(err_N(2)), casadi::MX::cos(err_N(2)))
	};
	casadi::MX err_N_w = casadi::MX::vertcat(en_vec);

	J += casadi::MX::mtimes({err_N_w.T(), P_cas, err_N_w});

	// Assemble and compile NLP
	casadi::MXDict nlp = {
		{"x", W},
		{"f", J},
		{"g", casadi::MX::vertcat(g)},
		{"p", P_param}
	};

	casadi::Dict opts;
	opts["ipopt.max_iter"]    = 100;
	opts["ipopt.tol"]         = 1e-4;
	opts["ipopt.print_level"] = 0;
	opts["print_time"]        = false;

	casadi_solver = casadi::nlpsol("solver", "ipopt", nlp, opts);
	solver_initialized = true;
}

void MPC::setObservabilityMatrix(const MatrixXd& A_k)
{
    O.resize(f*r, n);
    O.setZero();

    MatrixXd _temp = MatrixXd::Identity(n, n);
    for (int i = 0; i < f; i++)
    {
        _temp = _temp * A_k;
        O(seq(i*r, (i+1)*r-1), all) = C * _temp;
    }
}

void MPC::setToeplitzMatrix(const MatrixXd& A_k, const MatrixXd& B_k)
{
    M.resize(f*r, v*m);
    M.setZero();
    MatrixXd _temp;

    for (int i = 0; i < f; i++)
    {
        if (i < v)
        {
            for (int j = 0; j < i+1; j++)
            {
                if (j == 0)
                    _temp = MatrixXd::Identity(n, n);
                else
                    _temp = _temp * A_k;

                M(seq(i*r,(i+1)*r-1), seq((i-j)*m,(i-j+1)*m-1)) = C * _temp * B_k;
            }
        }
        else
        {
            for (int j = 0; j < v; j++)
            {
                if (j == 0)
                {
                    MatrixXd sumLast = MatrixXd::Zero(n, n);
                    _temp = MatrixXd::Identity(n, n);
                    for (int s = 0; s <= i; s++)
                    {
                        sumLast = sumLast + _temp;
                        _temp = _temp * A_k;
                    }
                    M(seq(i*r,(i+1)*r-1), seq((v-1)*m, v*m-1)) = C * sumLast * B_k;
                }
                else
                {
                    _temp = _temp * A_k;
                    M(seq(i*r,(i+1)*r-1), seq((v-1-j)*m,(v-j)*m-1)) = C * _temp * B_k;
                }
            }
        }
    }
}

std::tuple<MatrixXd,MatrixXd> MPC::getWeightMatrices(tuple<double, double, double> weights)
{
    // wt1: finite difference matrix for control smoothness
    MatrixXd wt1 = MatrixXd::Zero(v*m, v*m);
    for (int i = 0; i < v; i++)
    {
        wt1(seq(i*m,(i+1)*m-1), seq(i*m,(i+1)*m-1)) = MatrixXd::Identity(m, m);
        if (i > 0)
            wt1(seq(i*m,(i+1)*m-1), seq((i-1)*m,i*m-1)) = -MatrixXd::Identity(m, m);
    }

    // wt2: control rate weight — full m x m blocks on diagonal
    MatrixXd wt2 = MatrixXd::Zero(v*m, v*m);
    wt2(seq(0, m-1), seq(0, m-1)) = get<0>(weights) * MatrixXd::Identity(m, m);
    for (int i = 1; i < v; i++)
        wt2(seq(i*m,(i+1)*m-1), seq(i*m,(i+1)*m-1)) = get<1>(weights) * MatrixXd::Identity(m, m);

    MatrixXd W3 = wt1.transpose() * wt2 * wt1;

    // W4: tracking weight — full r x r blocks on diagonal
    MatrixXd W4 = MatrixXd::Zero(f*r, f*r);
    for (int i = 0; i < f; i++)
        W4(seq(i*r,(i+1)*r-1), seq(i*r,(i+1)*r-1)) = get<2>(weights) * MatrixXd::Identity(r, r);

    return std::make_tuple(W3, W4);
}

VectorXd MPC::nonlinearDynamics(const VectorXd& x, const VectorXd& u) const
{
    double theta = x(2);
    VectorXd x_dot(n);
    x_dot(0) = u(0)*cos(theta) - u(1)*sin(theta);
    x_dot(1) = u(0)*sin(theta) + u(1)*cos(theta);
    x_dot(2) = u(2);
    return x + sampling * x_dot;
}

std::tuple<MatrixXd,MatrixXd> MPC::linearizeModel(const VectorXd& x_bar, const VectorXd& u_bar) const
{
    double eps = 1e-5;
    MatrixXd A_lin(n, n), B_lin(n, m);
    VectorXd f0 = nonlinearDynamics(x_bar, u_bar);

    for (int i = 0; i < n; i++) {
        VectorXd x_pert = x_bar;
        x_pert(i) += eps;
        A_lin.col(i) = (nonlinearDynamics(x_pert, u_bar) - f0) / eps;
    }

    for (int i = 0; i < m; i++) {
        VectorXd u_pert = u_bar;
        u_pert(i) += eps;
        B_lin.col(i) = (nonlinearDynamics(x_bar, u_pert) - f0) / eps;
    }

    return {A_lin, B_lin};
}

void MPC::computeControlInputs()
{
	if(!solver_initialized) {
		std::cerr << "SOLVER NOT INITIALIZED" << std::endl;
		return;
	}

	VectorXd x_k = states.col(k);

	// Sync theta convention with reference
	double ref_theta_k = desiredInput(k, 2);
	while(x_k(2) - ref_theta_k >  M_PI) x_k(2) -= 2*M_PI;
	while(x_k(2) - ref_theta_k < -M_PI) x_k(2) += 2*M_PI;
	states(2, k) = x_k(2);

	MatrixXd refWindow = desiredInput(seq(k, k+f), all);

	std::vector<double> p_val;
	for(int i = 0; i < n; i++)
		p_val.push_back(x_k(i));
	for(int i = 0; i <= f; i++)
		for(int j = 0; j < n; j++)
			p_val.push_back(refWindow(i,j));

	std::vector<double> lbw, ubw, w0;
	for(int k_h = 0; k_h < f; k_h++)
	{
		for(int i = 0; i < n; i++) {
			lbw.push_back(-1e9);
			ubw.push_back( 1e9);
			w0.push_back(0.0);
		}
		for(int i = 0; i < m; i++) {
			lbw.push_back(u_min(i));
			ubw.push_back(u_max(i));
			w0.push_back(u_prev(i));
		}
	}
	for(int i = 0; i < n; i++) {
		lbw.push_back(-1e9);
		ubw.push_back( 1e9);
		w0.push_back(0.0);
	}

	int n_constraints = n * (f + 1);
	std::vector<double> lbg(n_constraints, 0.0);
	std::vector<double> ubg(n_constraints, 0.0);

	casadi::DMDict args = {
		{"x0",  casadi::DM(w0)},
		{"lbx", casadi::DM(lbw)},
		{"ubx", casadi::DM(ubw)},
		{"lbg", casadi::DM(lbg)},
		{"ubg", casadi::DM(ubg)},
		{"p",   casadi::DM(p_val)}
	};

	casadi::DMDict sol  = casadi_solver(args);
	std::vector<double> w_opt(sol.at("x"));

	VectorXd u_apply(m);
	for(int i = 0; i < m; i++)
		u_apply(i) = w_opt[n + i];

	inputs.col(k) = u_apply;
	u_prev        = u_apply;

	states.col(k+1) = nonlinearDynamics(x_k, u_apply);

	double ref_next = desiredInput(std::min(k+1, (unsigned int)desiredInput.rows()-1), 2);
	while(states(2,k+1) - ref_next >  M_PI) states(2,k+1) -= 2*M_PI;
	while(states(2,k+1) - ref_next < -M_PI) states(2,k+1) += 2*M_PI;

	outputs.col(k) = C * states.col(k);
	k++;
}

void MPC::writeToCSV(const std::string& filename, const MatrixXd& matrix) const {
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
    ofstream file(filename);
    if (file.is_open())
        file << matrix.format(CSVFormat);
    else
        cerr << "Error: Could not open " << filename << endl;
}

void MPC::saveData(string desiredInput_File, string input_File,
                   string state_File, string output_File,
                   string O_File, string M_File) const
{
    writeToCSV(desiredInput_File, desiredInput);
    writeToCSV(input_File,  inputs);
    writeToCSV(state_File,  states);
    writeToCSV(output_File, outputs);
    writeToCSV(O_File, O);
    writeToCSV(M_File, M);
}
