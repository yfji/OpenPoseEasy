// openpose.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "poseengine.h"
using namespace op;

int main()
{
	op::PoseEngine engine;
	cv::Mat image=cv::imread("ski.jpg");
	vector<float> keypoints;
	vector<int> keypointShape;

	engine.keypointsFromImage(image, keypoints, keypointShape);
	cout << "num people: " << keypointShape[0] << endl;
	cout << "num keypoints: " << keypointShape[1] << endl;
	cout << "num channels: " << keypointShape[2] << endl;
    return 0;
}

