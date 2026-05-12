#include "Controller.hpp"
#include "Observer.hpp"
#include "Pathfinder.hpp"
#include "network.hpp"
#include <Eigen/Dense>
#include <arpa/inet.h>
#include <casadi/core/integrator.hpp>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>

using namespace Eigen;
using Eigen::placeholders::all;

void writeToCSV(const std::string &filename, const MatrixXd &matrix) {
  const static IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols,
                                  ", ", "\n");
  std::ofstream file(filename);
  if (file.is_open())
    file << matrix.format(CSVFormat);
  else
    std::cerr << "Error: Could not open " << filename << std::endl;
}

int main() {

  int serverSocket = socket(AF_INET, SOCK_DGRAM, 0);

  if (serverSocket < 0) {
    std::cerr << "Socket creation failed\n";
    return 1;
  }

  // Server address
  sockaddr_in serverAddr{};
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_addr.s_addr = INADDR_ANY;
  serverAddr.sin_port = htons(8080);

  // Bind socket
  if (bind(serverSocket, (sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {

    std::cerr << "Bind failed\n";
    close(serverSocket);
    return 1;
  }
  char buffer[1024];

  sockaddr_in clientAddr{};
  socklen_t clientLen = sizeof(clientAddr);

  pthread_t tcp_thread;
  pthread_create(&tcp_thread, NULL, tcp_client, NULL);

  // Receive data
  int bytesReceived = recvfrom(serverSocket, buffer, sizeof(buffer) - 1, 0,
                               (sockaddr *)&clientAddr, &clientLen);

  if (bytesReceived < 0) {
    std::cerr << "Receive failed\n";
  }

  buffer[bytesReceived] = '\0';

  // Print received message
  std::cout << buffer << std::endl;


  // Parameters
  unsigned int f = 20;
  unsigned int v = 14;

  double sampling = 0.05;
  double cellSize = 0.05;
  double speed = 0.3;

  MatrixXd C = MatrixXd::Identity(3, 3);

  VectorXd x0(3);
  x0 << 0, 0, -0.2;

  VectorXd u_min(2), u_max(2);
  u_min << -1.2, -3.0;
  u_max << 1.2, 3.0;

  // Grid
  std::vector<std::vector<int>> grid = {
      {0, 1, 1, 0, 0}, 
			{0, 1, 1, 1, 0}, 
			{0, 0, 1, 1, 0}, 
			{0, 1, 1, 1, 0},
      {0, 1, 1, 1, 0}, 
			{0, 0, 0, 1, 0}, 
			{0, 1, 0, 0, 0}, 
			{0, 0, 0, 0, 0},
      {0, 1, 0, 0, 0}, 
			{0, 0, 1, 1, 0}, 
			{0, 1, 0, 1, 0}, 
			{0, 0, 0, 1, 0},
      {0, 1, 0, 1, 0}, 
			{0, 0, 1, 1, 0}, 
			{0, 1, 0, 1, 0}, 
			{0, 1, 0, 0, 0},
      {0, 0, 0, 1, 0}, 
			{0, 0, 0, 1, 0}, 
			{0, 1, 0, 1, 0}};


  double trackingWeight = 150.0;
  double controlWeight = 80.0;
  double omegaWeight = 2.0;
  double rateWeight = 800.0;

  auto horizons = std::make_tuple(v, f);
  auto weights =
      std::make_tuple(controlWeight, omegaWeight, trackingWeight, rateWeight);

  VectorXd x_current = x0;
  VectorXd x_true = x0;

  // MatrixXd log_states(3, simSteps);
  // MatrixXd log_inputs(2, simSteps);

  float goal_pose[2]={0.0,0.0};

  double alpha = 1.0;
  VectorXd u_prev = VectorXd::Zero(2);

	std::cout<<"Initial state:"<<std::endl;
	std::cin>>x_current[0]>>x_current[1]>>x_current[2];
	

  MPC mpc(C, horizons, weights,sampling, u_min, u_max);

	while(true){

		std::cout<<"Goal pose:"<<std::endl;
		std::cin>>goal_pose[0]>>goal_pose[1];

		Node start((int)x_current[0]/cellSize,(int)x_current[1]/cellSize);
		Node goal((int)goal_pose[0]/cellSize,(int)goal_pose[1]/cellSize);

		auto rawPath = FindPath(grid, start, goal);
		double pathLength = (rawPath.size() - 1) * cellSize;
		int simSteps = (int)(pathLength / speed / sampling);

		// std::cout << "Path size:  " << rawPath.size() << std::endl;
		// std::cout << "f:          " << f << std::endl;
		// std::cout << "simSteps:   " << simSteps << std::endl;
		// std::cout << "f + simSteps: " << f + simSteps << std::endl;

		if(rawPath.empty()){
			std::cerr<<"No path found!"<<std::endl;
			continue;
		}
		auto desiredTrajectory = getTrajectory(rawPath, simSteps + f,cellSize);
		mpc.setTrajectory(desiredTrajectory);
			for (int i = 0; i < simSteps; i++) {

			recv_data data = shared_recv.load(std::memory_order_relaxed);
			std::cout << "Received -> x: " << data.x << "  y: " << data.y << "  z: " << data.z
					  << std::endl;

			x_current(0) = data.x;
			x_current(1) = data.y;
			x_current(2) = data.z;

			VectorXd u_raw = mpc.computeControlInputs(x_current);
			VectorXd u = alpha * u_raw + (1.0 - alpha) * u_prev;
			u_prev = u;

			send_data s{};
			s.vx = u[0]; // placeholder
			s.w = u[1];  // placeholder
			shared_send.store(s, std::memory_order_relaxed);

			std::cout << "Sending  -> vx: " << s.vx << "  w: " << s.w << std::endl;
			std::string reply =
				std::to_string(u[0]) + "," + std::to_string(u[1]) + "\n";

			// Send response
			sendto(serverSocket, reply.c_str(), reply.size(), 0,
				   (sockaddr *)&clientAddr, clientLen);
		}
		printf("current pose: %f, %f, %f\n", x_current(0), x_current(1), x_current(2));
		mpc.resetTimestamp();
}

  // writeToCSV("data/states.csv", log_states);
  // writeToCSV("data/computedInputs.csv", log_inputs);
  // writeToCSV("data/trajectory.csv", desiredTrajectory);
  // std::cout << "Simulation completed!" << std::endl;
  return 0;
}
