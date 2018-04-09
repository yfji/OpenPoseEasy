#pragma once
#include <opencv2/core/core.hpp>
#include <vector>
using namespace std;

class HandDetector
{
public:
	HandDetector();
	virtual ~HandDetector();

public:
	cv::Mat* pImage;
public:
	virtual cv::Rect2f detectHand(vector<float>& armKeypoints, float thresh) = 0;
};

