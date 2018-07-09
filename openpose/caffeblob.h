#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

namespace op {
	template<typename Dtype>
	class CaffeBlob {
	public:
		Dtype* data;
		int count;
		int num;
		int channels;
		int height;
		int width;
		int capacity_count;
		bool data_self;

	public:
		CaffeBlob(int _num, int _channels, int _height, int _width) :
			num(_num),
			channels(_channels),
			height(_height),
			width(_width)
		{
			count = num*channels*height*width;
			capacity_count = count * sizeof(Dtype);
			data = new Dtype[count];
			data_self = true;
			for (auto i = 0; i < count; ++i) {
				data[i] = 0.0f;
			}
		}

		CaffeBlob() {
			data_self = false;
		}
		~CaffeBlob() {
			if (data_self)
				delete data;
		}
	public:

		void FromImage(const cv::Mat& image);
		cv::Mat ToImage();
	};
}

namespace op {
	template<typename Dtype>
	void CaffeBlob<Dtype>::FromImage(const cv::Mat & image)
	{
		Dtype* img_data = (Dtype*)image.data;
		if (data_self) {
			assert(count == image.channels()*image.cols*image.rows);
			for (auto i = 0; i < count; ++i) {
				data[i] = img_data[i];
			}
		}
		else {
			num = 1;
			channels = image.channels();
			height = image.rows;
			width = image.cols;
			count = num*channels*height*width;
			capacity_count = count * sizeof(Dtype);
			data = img_data;
		}
	}

	template<typename Dtype>
	cv::Mat CaffeBlob<Dtype>::ToImage()
	{
		vector<cv::Mat> image_channels;
		Dtype* cur_data = data;
		for (auto i = 0; i < channels; ++i) {
			cv::Mat chn(height, width, CV_32FC1, cur_data);
			image_channels.push_back(chn);
			cur_data += width*height;
		}
		cv::Mat image;
		cv::merge(image_channels, image);
		return image;
	}
}

