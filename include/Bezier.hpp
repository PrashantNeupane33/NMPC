#include <cmath>
#include <vector>
#include <utility>
#include <Eigen/Dense>

using Eigen::MatrixXd;

struct BezierSegment {
	double x0, y0;   // start
	double x1, y1;   // control point 1
	double x2, y2;   // control point 2
	double x3, y3;   // end
};

std::pair<double,double> bezierPoint(const BezierSegment& seg, double t)
{
	double mt  = 1.0 - t;
	double mt2 = mt  * mt;
	double mt3 = mt2 * mt;
	double t2  = t   * t;
	double t3  = t2  * t;

	double x = mt3*seg.x0 + 3*mt2*t*seg.x1 + 3*mt*t2*seg.x2 + t3*seg.x3;
	double y = mt3*seg.y0 + 3*mt2*t*seg.y1 + 3*mt*t2*seg.y2 + t3*seg.y3;
	return {x, y};
}

std::pair<double,double> bezierDerivative(const BezierSegment& seg, double t)
{
	double mt  = 1.0 - t;
	double mt2 = mt * mt;
	double t2  = t  * t;

	double dx = 3*mt2*(seg.x1-seg.x0) + 6*mt*t*(seg.x2-seg.x1) + 3*t2*(seg.x3-seg.x2);
	double dy = 3*mt2*(seg.y1-seg.y0) + 6*mt*t*(seg.y2-seg.y1) + 3*t2*(seg.y3-seg.y2);
	return {dx, dy};
}

std::vector<BezierSegment> nodesToBezierSegments(const std::vector<Node>& path, float cellSize)
{
    std::vector<BezierSegment> segments;

    for (int i = 0; i + 1 < (int)path.size(); i++)
    {
        double x0 = path[i].x     * cellSize;
        double y0 = path[i].y     * cellSize;
        double x3 = path[i+1].x   * cellSize;
        double y3 = path[i+1].y   * cellSize;

        // Control points: 1/3 and 2/3 along the straight line
        // (makes a smooth curve through all waypoints)
        double x1 = x0 + (x3 - x0) / 3.0;
        double y1 = y0 + (y3 - y0) / 3.0;
        double x2 = x0 + 2.0 * (x3 - x0) / 3.0;
        double y2 = y0 + 2.0 * (y3 - y0) / 3.0;

        segments.push_back({x0, y0, x1, y1, x2, y2, x3, y3});
    }

    return segments;
}

std::vector<BezierSegment> computeSmoothBezierSpline(const std::vector<Node>& path, float cellSize)
{
    int n = path.size() - 1;  // number of segments
    // n+1 waypoints → n segments

    // Convert nodes to doubles
    std::vector<double> px(n+1), py(n+1);
    for (int i = 0; i <= n; i++) {
        px[i] = path[i].x * cellSize;
        py[i] = path[i].y * cellSize;
    }

    // Solve tridiagonal system for control points
    // cp1 = first control point of each segment  (n values)
    // cp2 = second control point of each segment (n values, derived from cp1)
    std::vector<double> cp1x(n), cp1y(n);
    std::vector<double> cp2x(n), cp2y(n);

    // --- Solve for X ---
    std::vector<double> rhs(n);

    // Build right-hand side
    rhs[0] = px[0] + 2*px[1];
    for (int i = 1; i < n-1; i++)
        rhs[i] = 4*px[i] + 2*px[i+1];
    rhs[n-1] = (8*px[n-1] + px[n]) / 2.0;

    // Thomas algorithm (tridiagonal solver)
    std::vector<double> tmp(n);
    tmp[0] = 0.5;
    rhs[0] /= 2.0;
    for (int i = 1; i < n; i++) {
        double m = 1.0 / (4.0 - tmp[i-1]);
        tmp[i]   = m;
        rhs[i]   = (rhs[i] - rhs[i-1]) * m;
    }
    cp1x[n-1] = rhs[n-1];
    for (int i = n-2; i >= 0; i--)
        cp1x[i] = rhs[i] - tmp[i] * cp1x[i+1];

    // --- Solve for Y (same system) ---
    rhs[0] = py[0] + 2*py[1];
    for (int i = 1; i < n-1; i++)
        rhs[i] = 4*py[i] + 2*py[i+1];
    rhs[n-1] = (8*py[n-1] + py[n]) / 2.0;

    tmp[0] = 0.5;
    rhs[0] /= 2.0;
    for (int i = 1; i < n; i++) {
        double m = 1.0 / (4.0 - tmp[i-1]);
        tmp[i]   = m;
        rhs[i]   = (rhs[i] - rhs[i-1]) * m;
    }
    cp1y[n-1] = rhs[n-1];
    for (int i = n-2; i >= 0; i--)
        cp1y[i] = rhs[i] - tmp[i] * cp1y[i+1];

    // Derive cp2 from cp1 (mirror across waypoint)
    for (int i = 0; i < n-1; i++) {
        cp2x[i] = 2*px[i+1] - cp1x[i+1];
        cp2y[i] = 2*py[i+1] - cp1y[i+1];
    }
    cp2x[n-1] = (px[n] + cp1x[n-1]) / 2.0;
    cp2y[n-1] = (py[n] + cp1y[n-1]) / 2.0;

    // Build segments
    std::vector<BezierSegment> segments(n);
    for (int i = 0; i < n; i++) {
        segments[i] = {
            px[i],   py[i],    // start
            cp1x[i], cp1y[i],  // control point 1
            cp2x[i], cp2y[i],  // control point 2
            px[i+1], py[i+1]   // end
        };
    }

    return segments;
}

MatrixXd getBezierTrajectory(unsigned int timeSteps, double sampling, std::vector<BezierSegment> segments)
{
	// std::vector<BezierSegment> segments = {
	// 	{0.0,  1.0,   8.0,  1.0,  -8.0,  5.0,   0.0,  5.0},
	// 	{0.0,  5.0,   8.0,  5.0,  -8.0,  9.0,   0.0,  9.0},
	// 	{0.0,  9.0,   3.0, 11.0,   3.0,  7.0,   0.0,  7.0},
	// 	{0.0,  7.0,  -2.0,  5.0,   2.0,  3.0,   0.0,  0.0},
	// 	{0.0,  0.0,   5.0,  0.0,   5.0,  0.0,   5.0,  0.0},
	// 	{3.0,  7.0,   3.0,  7.0,   3.0,  5.0,   0.0,  5.0},
	// 	{0.0,  5.0,  -3.0,  5.0,   4.0,  9.0,   0.0,  9.0},
	// 	{-2.0, 9.0,   1.0,  9.0,   1.0,  6.0,  -2.0,  6.0},
	// 	{-2.0, 6.0,  -5.0,  6.0,   3.0,  3.0,  -3.0,  3.0},
	// 	{-3.0, 3.0,  -6.0,  3.0,  -6.0,  7.0,  -4.0,  7.0},
	// 	{5.0,  0.0,   5.0,  0.0,   5.0,  3.0,   2.0,  3.0},
	// 	{-4.0, 7.0,  -4.0,  7.0,  -4.0,  4.0,  -7.0,  4.0},
	// 	{-7.0, 4.0,  -7.0,  4.0,  -7.0,  8.0,  -4.0,  8.0},
	// 	{-4.0, 8.0,  -4.0,  8.0,  -4.0, 11.0,  -7.0, 11.0},
	// 	{-7.0,11.0,  -7.0, 11.0,  -7.0,  1.0,   0.0,  1.0},
	// 	{0.0,  9.0,  -4.0,  9.0,   3.0, 12.0,  -2.0, 12.0},
	// 	{-2.0,12.0,  -5.0, 12.0,  -5.0,  9.0,  -2.0,  9.0},
	// 	{2.0,  3.0,  -1.0,  3.0,   6.0,  6.0,   3.0,  6.0},
	// 	{3.0,  6.0,   3.0,  6.0,   3.0,  4.0,   6.0,  4.0},
	// 	{6.0,  4.0,   6.0,  4.0,   6.0,  7.0,   3.0,  7.0},
	// };
	int nSeg = segments.size();
	std::vector<double> segLen(nSeg);
	int arcSamples = 100;

	for(int s = 0; s < nSeg; s++)
	{
		double len = 0;
		auto prev  = bezierPoint(segments[s], 0.0);
		for(int j = 1; j <= arcSamples; j++)
		{
			double t    = j / (double)arcSamples;
			auto   curr = bezierPoint(segments[s], t);
			double dx   = curr.first  - prev.first;
			double dy   = curr.second - prev.second;
			len        += sqrt(dx*dx + dy*dy);
			prev        = curr;
		}
		segLen[s] = len;
	}

	// Cumulative length
	std::vector<double> cumLen(nSeg + 1, 0.0);
	for(int s = 0; s < nSeg; s++)
		cumLen[s+1] = cumLen[s] + segLen[s];

	double totalLen = cumLen[nSeg];
	double speed    = 0.8;
	double ds       = speed * sampling;

	MatrixXd traj;
	traj.resize(timeSteps, 3);

	for(int i = 0; i < timeSteps; i++)
	{
		double arcPos = fmod(i * ds, totalLen);

		// Find segment
		int seg = 0;
		for(int s = 0; s < nSeg; s++) {
			if(arcPos >= cumLen[s] && arcPos < cumLen[s+1]) {
				seg = s;
				break;
			}
		}

		// t within segment, normalized by arc length
		double t = (arcPos - cumLen[seg]) / segLen[seg];
		t = std::max(0.0, std::min(1.0, t));

		auto pos = bezierPoint(segments[seg], t);
		auto der = bezierDerivative(segments[seg], t);

		traj(i, 0) = pos.first;
		traj(i, 1) = pos.second;
		traj(i, 2) = atan2(der.second, der.first);
	}
	return traj;
}
