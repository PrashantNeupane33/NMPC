#include "Pathfinder.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <climits>

Node::Node(int _x, int _y)
    : x(_x) , y(_y) , f(0) , g(0) , h(0)
{
}

// Compare nodes by total cost (for priority queue)
bool Node::operator>(const Node& other) const
{
    return f > other.f;
}

// Equality check (same coordinates)
bool Node::operator==(const Node& other) const
{
    return x == other.x && y == other.y;
}

std::vector<Node> FindPath(const std::vector<std::vector<int>>& graph, const Node& start, const Node& goal)
{
    // Possible movement directions: up, right, down, left
    const int directionX[] = {-1, 0, 1, 0};
    const int directionY[] = {0, 1, 0, -1};

    int rows = graph.size();
    int cols = graph[0].size();

    // Priority queue (min-heap) sorted by lowest f cost
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openList;

    // Closed list: marks visited nodes
    std::vector<std::vector<bool>> closedList(rows, std::vector<bool>(cols, false));

    // gScore: best-known cost to reach each cell
    std::vector<std::vector<int>> gScore(rows, std::vector<int>(cols, INT_MAX));

    // Parent matrix: stores the node from which we arrived at each position
    std::vector<std::vector<Node>> parent(rows, std::vector<Node>(cols));

    // Initialize start node cost
    gScore[start.x][start.y] = 0;
    openList.push(start);

    // Main A* search loop
    while (!openList.empty())
    {
        // Get node with lowest f score
        Node current = openList.top();
        openList.pop();

        // Check if we've reached the goal
        if (current == goal)
        {
            // Reconstruct the path by backtracking through parents
            std::vector<Node> path;
            while (!(current == start))
            {
                path.push_back(current);
                current = parent[current.x][current.y];
            }
            path.push_back(start);

            // Reverse to get path from start → goal
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Mark current node as closed (visited)
        closedList[current.x][current.y] = true;

        // Explore all 4 neighboring cells
        for (int i = 0; i < 4; ++i)
        {
            int newX = current.x + directionX[i];
            int newY = current.y + directionY[i];

            // Check grid boundaries and walkability
            if (newX >= 0 && newX < rows && newY >= 0 && newY < cols && graph[newX][newY] == 0)
            {
                // Skip already processed (closed) cells
                if (closedList[newX][newY])
                    continue;

                // Tentative g cost (current cost + 1 for movement)
                int newG = gScore[current.x][current.y] + 1;

                // If we found a better path to this neighbor
                if (newG < gScore[newX][newY])
                {
                    gScore[newX][newY] = newG;

                    // Compute new neighbor costs
                    Node neighbor(newX, newY);
                    neighbor.g = newG;
                    neighbor.h = std::abs(newX - goal.x) + std::abs(newY - goal.y); // Manhattan distance
                    neighbor.f = neighbor.g + neighbor.h;

                    // Record parent (for path reconstruction)
                    parent[newX][newY] = current;

                    // Add neighbor to open list for further exploration
                    openList.push(neighbor);
                }
            }
        }
    }

    return {};// No path found 
}

// Intepolated path, vector-> Matrix
MatrixXd getTrajectory(const std::vector<Node>& path, int timeSteps, float cellSize){
    MatrixXd desiredTrajectory(timeSteps, 3);
    for(int i = 0; i < timeSteps; i++){
        double t    = (double)i / (timeSteps - 1) * (path.size() - 1);
        int    idx  = (int)t;
        double frac = t - idx;
        int    idx1 = std::min(idx + 1, (int)path.size() - 1);

        desiredTrajectory(i, 0) = (path[idx].x + frac * (path[idx1].x - path[idx].x))*cellSize;
        desiredTrajectory(i, 1) = (path[idx].y + frac * (path[idx1].y - path[idx].y))*cellSize;
        if(i < timeSteps - 1)
            desiredTrajectory(i, 2) = atan2(path[idx1].y - path[idx].y,
                                            path[idx1].x - path[idx].x);
        else
            desiredTrajectory(i, 2) = desiredTrajectory(i-1, 2);
    }
    return desiredTrajectory;
}

void PrintPath(const std::vector<Node>& path)
{
    for (const Node& node : path)
    {
        std::cout << "(" << node.x << ", " << node.y << ") ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}
