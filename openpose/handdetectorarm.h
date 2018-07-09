#pragma once
#include "handdetector.h"
class HandDetectorArm :
	public HandDetector
{
public:
	HandDetectorArm();
	virtual ~HandDetectorArm();

public:
	virtual cv::Rect2f detectHand(vector<float>& armKeypoints, float thresh);
};

