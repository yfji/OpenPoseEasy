#include "stdafx.h"
#include "handdetectorarm.h"


HandDetectorArm::HandDetectorArm()
{
}


HandDetectorArm::~HandDetectorArm()
{
}

cv::Rect2f HandDetectorArm::detectHand(vector<float>& armKeypoints, float thresh)
{
	float wristScore = armKeypoints[2];
	float elbowScore = armKeypoints[5];
	float shoulderScore = armKeypoints[8];
	//cout << wristScore << "," << elbowScore << "," << shoulderScore << endl;
	if (wristScore < thresh || elbowScore < thresh || shoulderScore < thresh)
		return cv::Rect2f(0, 0, 0, 0);
	cv::Rect2f hand;
	hand.x = armKeypoints[0] + 0.3f*(armKeypoints[0] - armKeypoints[3]);
	hand.y = armKeypoints[1] + 0.3f*(armKeypoints[1] - armKeypoints[4]);
	float distWE = sqrt((armKeypoints[0] - armKeypoints[3])*(armKeypoints[0] - armKeypoints[3]) + (armKeypoints[1] - armKeypoints[4])*(armKeypoints[1] - armKeypoints[4]));
	float distES = sqrt((armKeypoints[3] - armKeypoints[6])*(armKeypoints[3] - armKeypoints[6]) + (armKeypoints[4] - armKeypoints[7])*(armKeypoints[4] - armKeypoints[7]));
	hand.width = 1.5f*max(distWE, 0.9f*distES);
	hand.height = hand.width;
	hand.x -= hand.width*0.5f;
	hand.y -= hand.height*0.5f;
	hand.x = max(0.0f, hand.x);
	hand.y = max(0.0f, hand.y);
	return hand;
}
