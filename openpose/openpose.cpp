// openpose.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "poseengine.h"
#include "handengine.h"
using namespace op;

int main()
{
	op::PoseEngine poseEngine;
	op::HandEngine handEngine;
	cv::Mat image=cv::imread("ski.jpg");
	char key = 0;
	cv::VideoCapture capture(0);
	cv::Mat frame;
	cv::Mat canvas;
	while (1) {
		capture >> frame;
		if (frame.empty())
			break;
		vector<float> keypoints;
		vector<int> keypointShape;

		canvas=poseEngine.keypointsFromImage(frame, keypoints, keypointShape);
		//cout << "num people: " << keypointShape[0] << endl;
		//cout << "num keypoints: " << keypointShape[1] << endl;
		//cout << "num channels: " << keypointShape[2] << endl;
		canvas = handEngine.handKeypointsFromImage(canvas, keypoints, keypointShape);
		cv::imshow("kp", canvas);
		key = cv::waitKey(10);
		if (key == 27)
			break;
	}
    return 0;
}

