#include<tuple>
#include<Eigen/Dense>
#include <casadi/casadi.hpp>
#include "Controller.hpp"

using namespace Eigen;
using namespace std;
using Eigen::placeholders::all;

MPC::MPC(MatrixXd _C,
         std::tuple<unsigned int, unsigned int> horizons,
         std::tuple<double, double, double> _weights,
         VectorXd _initialState,
         MatrixXd _desiredTrajectory,
         double _sampling,
         VectorXd _u_min,
         VectorXd _u_max):
    C(_C),
    f(get<1>(horizons)), v(get<0>(horizons)),
	weights(_weights),
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

    inputs.resize(m, maxSimulationSamples - 1);
    inputs.setZero();

    outputs.resize(r, maxSimulationSamples - 1);
    outputs.setZero();

    u_prev = VectorXd::Zero(m);

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
	Q_cas(0,0) = get<2>(weights);
	Q_cas(1,1) = get<2>(weights);
	Q_cas(2,2) = 1.0;

	casadi::MX R_cas = casadi::MX::zeros((int)m, (int)m);
	R_cas(0,0) = get<0>(weights);
	R_cas(1,1) = get<0>(weights);
	R_cas(2,2) = get<1>(weights);

	casadi::MX P_cas = casadi::MX::zeros((int)n, (int)n);
	P_cas(0,0) = get<2>(weights) * 2.0;
	P_cas(1,1) = get<2>(weights) * 2.0;
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

		J += casadi::MX::mtimes(std::vector<casadi::MX>{err_wrapped.T(), Q_cas, err_wrapped})
		   + casadi::MX::mtimes(std::vector<casadi::MX>{Uk.T(), R_cas, Uk});

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

	J += casadi::MX::mtimes(std::vector<casadi::MX>{err_N_w.T(), P_cas, err_N_w});

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

// std::tuple<MatrixXd,MatrixXd> MPC::getWeightMatrices(tuple<double, double, double> weights)
// {
//     // wt1: finite difference matrix for control smoothness
//     MatrixXd wt1 = MatrixXd::Zero(v*m, v*m);
//     for (int i = 0; i < v; i++)
//     {
//         wt1(seq(i*m,(i+1)*m-1), seq(i*m,(i+1)*m-1)) = MatrixXd::Identity(m, m);
//         if (i > 0)
//             wt1(seq(i*m,(i+1)*m-1), seq((i-1)*m,i*m-1)) = -MatrixXd::Identity(m, m);
//     }
//
//     // wt2: control rate weight
//     MatrixXd wt2 = MatrixXd::Zero(v*m, v*m);
//     wt2(seq(0, m-1), seq(0, m-1)) = get<0>(weights) * MatrixXd::Identity(m, m);
//     for (int i = 1; i < v; i++)
//         wt2(seq(i*m,(i+1)*m-1), seq(i*m,(i+1)*m-1)) = get<1>(weights) * MatrixXd::Identity(m, m);
//
//     MatrixXd W3 = wt1.transpose() * wt2 * wt1;
//
//     // W4: tracking weight
//     MatrixXd W4 = MatrixXd::Zero(f*r, f*r);
//     for (int i = 0; i < f; i++)
//         W4(seq(i*r,(i+1)*r-1), seq(i*r,(i+1)*r-1)) = get<2>(weights) * MatrixXd::Identity(r, r);
//
//     return std::make_tuple(W3, W4);
// }

VectorXd MPC::computeControlInputs(VectorXd x_k)
{
	while(x_k(2) >  M_PI) x_k(2) -= 2*M_PI;
	while(x_k(2) < -M_PI) x_k(2) += 2*M_PI;

	MatrixXd refWindow = desiredInput(seq(k, k+f), all);

	std::vector<double> p_val;
	for(int i = 0; i < n; i++)
		p_val.push_back(x_k(i));
	for(int i = 0; i <= f; i++)
		for(int j = 0; j < n; j++){
			double val = refWindow(i, j);
			if(j == 2)
				val = atan2(sin(val), cos(val));
			p_val.push_back(val);
		}

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
		{"x0", casadi::DM(w0)},
		{"lbx", casadi::DM(lbw)},
		{"ubx", casadi::DM(ubw)},
		{"lbg", casadi::DM(lbg)},
		{"ubg", casadi::DM(ubg)},
		{"p", casadi::DM(p_val)}
	};

	casadi::DMDict sol  = casadi_solver(args);
	std::vector<double> w_opt(sol.at("x"));

	VectorXd u_apply(m);
	for(int i = 0; i < m; i++)
		u_apply(i) = w_opt[n + i];

	inputs.col(k) = u_apply;
	u_prev = u_apply;
	k++;

	return u_apply;
}

