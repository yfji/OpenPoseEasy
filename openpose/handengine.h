#pragma once
#include "handdetectorarm.h"
#include "handdetectorwrist.h"
#include "caffewrapper.h"
#include <array>

#define HAND_PAIRS_RENDER_GPU 0,1,1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20
#define HAND_COLORS_RENDER_GPU \
        100.f,  100.f,  100.f, \
        100.f,    0.f,    0.f, \
        150.f,    0.f,    0.f, \
        200.f,    0.f,    0.f, \
        255.f,    0.f,    0.f, \
        100.f,  100.f,    0.f, \
        150.f,  150.f,    0.f, \
        200.f,  200.f,    0.f, \
        255.f,  255.f,    0.f, \
          0.f,  100.f,   50.f, \
          0.f,  150.f,   75.f, \
          0.f,  200.f,  100.f, \
          0.f,  255.f,  125.f, \
          0.f,   50.f,  100.f, \
          0.f,   75.f,  150.f, \
          0.f,  100.f,  200.f, \
          0.f,  125.f,  255.f, \
        100.f,    0.f,  100.f, \
        150.f,    0.f,  150.f, \
        200.f,    0.f,  200.f, \
        255.f,    0.f,  255.f

namespace op {
	class HandEngine
	{
	public:
		HandEngine();
		~HandEngine();

	private:
		std::shared_ptr<CaffeWrapper<float>> wrapper;
		std::shared_ptr<HandDetector> handDetector;

		std::vector<std::array<cv::Rect2f, 2>> handRectangles;
		const std::vector<unsigned int> HAND_PAIRS_RENDER{ HAND_PAIRS_RENDER_GPU };
		const std::vector<float> HAND_COLORS_RENDER{ HAND_COLORS_RENDER_GPU };
		const int hand_num_parts = 21;
		const int hand_max_people = 10;	//must be GE 2
		cv::Mat affineMatrix;
		cv::Mat* pImage;
		cv::Rect2f getHandRectangleByPose(vector<float>& armKeypoints, float thresh=0.05);
		
		std::shared_ptr<CaffeBlob<float>> input_blob;
		std::shared_ptr<CaffeBlob<float>> peak_blob;

	public:
		void getHandRectangles(const vector<float>& poseKeypoints, const vector<int>& shape);
		//void detectHands();
		cv::Mat cropFrame(cv::Rect& handRect, cv::Mat& im, const bool mirror=false);
		void findPeaks(CaffeBlob<float>* heatmaps, CaffeBlob<float>* out, float thresh = 0.05);
		void connectKeypoints(float* handKeypoints);
		void detectHandKeypoints(cv::Mat& hand_im);
		void renderHandKeypointsCpu(Mat& frame,
			CaffeBlob<float>* blobKeypoints,
			const float renderThreshold,
			float scale_x=1.0,
			float scale_y=1.0);

		void handKeypointsFromImage(cv::Mat& im, cv::Mat& canvas, const vector<float>& poseKeypoints, const vector<int>& shape);
	};

}

