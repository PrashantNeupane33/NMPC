#pragma once
#include <vector>
#include <queue>
#include <Eigen/Dense>
using Eigen::MatrixXd;

struct Node
{
    int x, y; // Coordinates
    int f, g, h; // Cost values

    Node(int _x = 0, int _y = 0);

    // For priority queue
    bool operator>(const Node& other) const;
    bool operator==(const Node& other) const;
};

// Finds a path from start to goal using the A* algorithm.
// @graph should be a 2D grid where 0 = walkable, 1 = blocked.
std::vector<Node> FindPath(const std::vector<std::vector<int>>& graph, const Node& start, const Node& goal);
MatrixXd getTrajectory(const std::vector<Node>& path, int timeSteps);

// Function to print the path
void PrintPath(const std::vector<Node>& path);
