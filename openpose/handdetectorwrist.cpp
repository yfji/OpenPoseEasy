#include "stdafx.h"
#include "handdetectorwrist.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

HandDetectorWrist::HandDetectorWrist()
{
}


HandDetectorWrist::~HandDetectorWrist()
{
}

//Region Growth
cv::Rect2f HandDetectorWrist::detectHand(vector<float>& armKeypoints, float thresh)
{
	float seedX = armKeypoints[0];
	float seedY = armKeypoints[1];
	float wristScore = armKeypoints[2];

	if(wristScore<thresh)
		return cv::Rect2f(0, 0, 0, 0);
	float ratio = 1.0*target_side / pImage->cols;
	if (pImage->cols < pImage->rows)
		ratio = 1.0*target_side / pImage->rows;
	cv::Mat temp=*pImage;
	scale = ratio;
	if (scale < 1) {
		cv::resize(*pImage, temp, cv::Size(), ratio, ratio, cv::INTER_CUBIC);
	}
	else {
		scale = 1.0;
	}
	//cv::imshow("temp", temp);
	regionGrowRGB(temp, (int)(seedX*scale), (int)(seedY*scale));
	return cv::Rect2f(handRect);
}

int HandDetectorWrist::calcDistance(int v1, int v2) {
	return abs(v1 - v2);
}
int HandDetectorWrist::calcDistanceRGB(cv::Vec3i& c1, cv::Vec3i& c2, cv::Vec3f& weights) {
	float sum = 0.0;
	for (auto i = 0; i < 3; ++i) {
		sum += 1.0*abs(c1[i] - c2[i])*weights[i];
	}
	return (int)sum;
}
void HandDetectorWrist::regionGrow(cv::Mat& img, int seedX, int seedY) {
	uchar* data = img.data;
	int w = img.cols;
	int h = img.rows;
	vector<int> neighbors = { -w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1 };

	int top = -1;
	int ltx = 1e4;
	int lty = 1e4;
	int rbx = 0;
	int rby = 0;

	memset(used, 0, sizeof(used));
	stack[++top] = seedY*w + seedX;
	int seed_value = data[seedY*w + seedX];
	while (top >= 0 && top < MAX_N) {
		int top_index = stack[top];
		char push = 0;
		for (auto i = 0; i < neighbors.size(); ++i) {
			int cur_index = top_index + neighbors[i];
			int cur_value = (int)data[cur_index];
			int r = cur_index / w;
			int c = cur_index - r*w;
			if (!used[cur_index] && r >= 0 && r < h && c >= 0 && c < w && calcDistance(seed_value, cur_value) < thresh) {
				used[cur_index] = 1;
				stack[++top] = cur_index;
				push = 1;
				ltx = min(c, ltx);
				rbx = max(c, rbx);
				lty = min(r, lty);
				rby = max(r, rby);
			}
		}
		if (!push)
			--top;
	}
	handRect.x = (int)(ltx*1.0 / scale);
	handRect.y = (int)(lty*1.0 / scale);
	handRect.width = (int)(1.0*(rbx - ltx) / scale);
	handRect.height = (int)(1.0*(rby - lty) / scale);

	//cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC1);
	//uchar* ptr_canvas = canvas.data;
	//for (auto i = 0; i < MAX_N; ++i) {
	//	if (used[i])
	//		ptr_canvas[i] = 255;
	//}
	//cv::imshow("mask", canvas);
}

void HandDetectorWrist::regionGrowRGB(cv::Mat& img, int seedX, int seedY) {
	uchar* data = img.data;
	int w = img.cols;
	int h = img.rows;
	int c = img.channels();
	vector<int> neighbors = { -w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1 };

	int up = w*h;
	int top = -1;
	int seedIndex = seedY*w + seedX;
	memset(used, 0, sizeof(used));
	stack[++top] = seedIndex;
	cv::Vec3i color_seed((int)data[seedIndex * 3], (int)data[seedIndex * 3 + 1], (int)data[seedIndex * 3 + 2]);
	float ratio_b = color_seed[2] * 1.0 / (color_seed[0] + color_seed[1] + color_seed[2]);
	float ratio_g = color_seed[1] * 1.0 / color_seed[2] * ratio_b;
	float ratio_r = color_seed[0] * 1.0 / color_seed[2] * ratio_b;
	cv::Vec3f seed_weights(ratio_b, ratio_g, ratio_r);
	int ltx = 1e4;
	int lty = 1e4;
	int rbx = 0;
	int rby = 0;
	while (top >= 0 && top < MAX_N) {
		int top_index = stack[top];
		char push = 0;
		for (auto i = 0; i < neighbors.size(); ++i) {
			int cur_index = top_index + neighbors[i];
			if (cur_index < 0 || cur_index >= up)
				continue;
			cv::Vec3i color_cur((int)data[cur_index * 3], (int)data[cur_index * 3 + 1], (int)data[cur_index * 3 + 2]);
			int r = cur_index / w;
			int c = cur_index - r*w;
			if (!used[cur_index] && r >= 0 && r < h && c >= 0 && c < w && calcDistanceRGB(color_seed, color_cur, seed_weights) < thresh) {
				used[cur_index] = 1;
				stack[++top] = cur_index;
				push = 1;
				ltx = min(c, ltx);
				rbx = max(c, rbx);
				lty = min(r, lty);
				rby = max(r, rby);
			}
		}
		if (!push)
			--top;
	}
	handRect.x = (int)(ltx*1.0 / scale);
	handRect.y = (int)(lty*1.0 / scale);
	handRect.width = (int)(1.0*(rbx - ltx) / scale);
	handRect.height = (int)(1.0*(rby - lty) / scale);
	handRect.x = max(0, handRect.x - handRect.width / 2);
	handRect.y = max(0, handRect.y - handRect.height / 2);
	handRect.width = min(pImage->cols - handRect.x, handRect.width * 2);
	handRect.height = min(pImage->rows - handRect.y, handRect.height * 2);

	cv::Mat handImg = pImage->clone();
	cv::rectangle(handImg, handRect, cv::Scalar(0, 0, 255), 2);
	cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC1);
	uchar* ptr_canvas = canvas.data;
	for (auto i = 0; i < MAX_N; ++i) {
		if (used[i])
			ptr_canvas[i] = 255;
	}
	cv::imshow("rect", handImg);
	cv::imshow("mask", canvas);
}
