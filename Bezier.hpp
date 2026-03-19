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

MatrixXd getTrajectory(unsigned int timeSteps, double sampling)
{
	std::vector<BezierSegment> segments = {
	{2.0,  3.0,  -1.0,  3.0,   6.0,  6.0,   3.0,  6.0},
	{3.0,  6.0,   3.0,  6.0,   3.0,  4.0,   6.0,  4.0},
	{6.0,  4.0,   6.0,  4.0,   6.0,  7.0,   3.0,  7.0},
	{3.0,  7.0,   3.0,  7.0,   3.0,  5.0,   0.0,  5.0},
	{0.0,  5.0,  -3.0,  5.0,   4.0,  9.0,   0.0,  9.0},
	{0.0,  9.0,  -4.0,  9.0,   3.0, 12.0,  -2.0, 12.0},
	{-2.0,12.0,  -5.0, 12.0,  -5.0,  9.0,  -2.0,  9.0},
	{-2.0, 9.0,   1.0,  9.0,   1.0,  6.0,  -2.0,  6.0},
	{-2.0, 6.0,  -5.0,  6.0,   3.0,  3.0,  -3.0,  3.0},
	{-3.0, 3.0,  -6.0,  3.0,  -6.0,  7.0,  -4.0,  7.0},
	{-4.0, 7.0,  -4.0,  7.0,  -4.0,  4.0,  -7.0,  4.0},
	{-7.0, 4.0,  -7.0,  4.0,  -7.0,  8.0,  -4.0,  8.0},
	{-4.0, 8.0,  -4.0,  8.0,  -4.0, 11.0,  -7.0, 11.0},
	{-7.0,11.0,  -7.0, 11.0,  -7.0,  1.0,   0.0,  1.0},
	{0.0,  1.0,   8.0,  1.0,  -8.0,  5.0,   0.0,  5.0},
	{0.0,  5.0,   8.0,  5.0,  -8.0,  9.0,   0.0,  9.0},
	{0.0,  9.0,   3.0, 11.0,   3.0,  7.0,   0.0,  7.0},
	{0.0,  7.0,  -2.0,  5.0,   2.0,  3.0,   0.0,  0.0},
	{0.0,  0.0,   5.0,  0.0,   5.0,  0.0,   5.0,  0.0},
	{5.0,  0.0,   5.0,  0.0,   5.0,  3.0,   2.0,  3.0},
};
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
// Unwrap theta
	for(int i = 1; i < timeSteps; i++)
	{
		double diff = traj(i,2) - traj(i-1,2);
		if      (diff >  M_PI) traj(i,2) -= 2*M_PI;
		else if (diff < -M_PI) traj(i,2) += 2*M_PI;
	}
	return traj;
}
