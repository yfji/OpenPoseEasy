#pragma once
#include "handdetector.h"

#define MAX_N	90000
class HandDetectorWrist :
	public HandDetector
{
public:
	HandDetectorWrist();
	virtual ~HandDetectorWrist();

private:
	int stack[MAX_N];
	char used[MAX_N];
	int top;
	const int thresh = 20;
	const int target_side = 300;
	float scale;
	cv::Rect handRect;
private:
	int calcDistance(int v1, int v2);
	int calcDistanceRGB(cv::Vec3i& c1, cv::Vec3i& c2, cv::Vec3f& weights);
	void regionGrow(cv::Mat& img, int seedX, int seedY);
	void regionGrowRGB(cv::Mat& img, int seedX, int seedY);
public:
	virtual cv::Rect2f detectHand(vector<float>& armKeypoints, float thresh);
};

