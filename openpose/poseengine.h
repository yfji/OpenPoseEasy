#pragma once#
#include "caffewrapper.h"

#define POSE_COCO_COLORS_RENDER_GPU \
	255.f, 0.f, 85.f, \
	255.f, 0.f, 0.f, \
	255.f, 85.f, 0.f, \
	255.f, 170.f, 0.f, \
	255.f, 255.f, 0.f, \
	170.f, 255.f, 0.f, \
	85.f, 255.f, 0.f, \
	0.f, 255.f, 0.f, \
	0.f, 255.f, 85.f, \
	0.f, 255.f, 170.f, \
	0.f, 255.f, 255.f, \
	0.f, 170.f, 255.f, \
	0.f, 85.f, 255.f, \
	0.f, 0.f, 255.f, \
	255.f, 0.f, 170.f, \
	170.f, 0.f, 255.f, \
	255.f, 0.f, 255.f, \
	85.f, 0.f, 255.f

namespace op {
	class PoseEngine
	{
	public:
		PoseEngine();
		~PoseEngine();

	public:
		std::shared_ptr<CaffeWrapper<float>> wrapper;
		const std::vector<float> POSE_COCO_COLORS_RENDER{ POSE_COCO_COLORS_RENDER_GPU };
		const std::vector<unsigned int> POSE_COCO_PAIRS_RENDER{ 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17 };
		const unsigned int POSE_MAX_PEOPLE = 96;
	
	public:
		void connectBodyPartsCpu(vector<float>& poseKeypoints,
			const float* const heatMapPtr,
			const float* const peaksPtr,
			const Size& heatMapSize,
			const int maxPeaks,
			const int interMinAboveThreshold,
			const float interThreshold,
			const int minSubsetCnt,
			const float minSubsetScore,
			const float scaleFactor,
			vector<int>& keypointShape);

		void nms(CaffeBlob<float> *bottom_blob, CaffeBlob<float> *top_blob, float threshold);
		void renderKeypointsCpu(Mat& frame,
			const vector<float>& keypoints,
			vector<int> keyshape,
			const std::vector<unsigned int>& pairs,
			const std::vector<float> colors,
			const float thicknessCircleRatio,
			const float thicknessLineRatioWRTCircle,
			const float threshold, float scale_x, float scale_y);

		void renderPoseKeypointsCpu(Mat& frame,
			const vector<float>& poseKeypoints,
			vector<int> keyshape,
			const float renderThreshold,
			float scale_x,
			float scale_y,
			const bool blendOriginalFrame = true);

		void renderKeypointsOnly(Mat& frame,
			const vector<float>& poseKeypoints,
			vector<int> keyshape,
			const std::vector<float> colors,
			const float renderThreshold,
			float scale_x,
			float scale_y,
			const bool blendOriginalFrame = true);
		
		cv::Mat keypointsFromImage(cv::Mat& im, vector<float>& keypoints, vector<int>& keypointShape);
	
	private:		
		inline float fastMax(float a, float b) {
			return a > b ? a : b;
		}
		inline float fastMin(float a, float b) {
			return a < b ? a : b;
		}
		inline int intRound(float f) {
			return int(f + 0.5);
		}
	};
};

