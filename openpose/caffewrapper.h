#pragma once
#include <caffe/caffe.hpp>
#include <caffe/layers/input_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/dropout_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/concat_layer.hpp>
#include <caffe/blob.hpp>
#include "caffeblob.h"
#include <iostream>
#include <vector>
#include <memory>
using namespace std;
using namespace caffe;
using namespace cv;

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	//REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	//REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	//REGISTER_LAYER_CLASS(Pooling);

}

namespace op {
	template<typename Dtype>
	class CaffeWrapper
	{
	public:
		CaffeWrapper(const string& prototxt_file,
			const string& caffemodel_file,
			const string& mean_file = "");
		~CaffeWrapper();

	public:
		const int im_w = 656;
		const int im_h = 368;
		float scale_x;
		float scale_y;

		void reshape(const int width, const int height);
		cv::Mat reshapeImage(const cv::Mat& image);
		void forwardImage(const string& input_name, cv::Mat& im);
		void forwardBlob(const string& input_name, CaffeBlob<Dtype>* net_input);
		void getOutputBlob(const string& output_name, CaffeBlob<Dtype>* net_output);

	private:
		int num_channels_;
		std::shared_ptr<Net<Dtype>> net_;
		cv::Size input_size_;
	private:
		cv::Mat getImagePadResize(cv::Mat& img, const cv::Size& baseSize);
		cv::Mat getImageResize(cv::Mat& img, const cv::Size& baseSize);
	};
}

namespace op {
	template<typename Dtype>
	CaffeWrapper<Dtype>::CaffeWrapper(const string& prototxt_file,
		const string& caffemodel_file,
		const string& mean_file = "")
	{
		Caffe::set_mode(Caffe::GPU);

		/* Load the network. */
		net_.reset(new Net<float>(prototxt_file, caffe::TEST));
		net_->CopyTrainedLayersFrom(caffemodel_file);

		CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
		Blob<float>* input_layer = net_->input_blobs()[0];
		num_channels_ = input_layer->channels();
		CHECK(num_channels_ == 3 || num_channels_ == 1)
			<< "Input layer should have 1 or 3 channels.";
		input_size_ = cv::Size(input_layer->width(), input_layer->height());
		if (input_size_.width == 1 && input_size_.height == 1) {	///adjust dynamically
			input_layer->Reshape(1, num_channels_, im_h, im_w);
			net_->Reshape();
		}
		else {
			CHECK_EQ(input_size_.width, im_w) << "Network width must be " << im_w;
			CHECK_EQ(input_size_.height, im_h) << "Network height must be " << im_h;
		}
	}

	template<typename Dtype>
	CaffeWrapper<Dtype>::~CaffeWrapper()
	{
	}

	template<typename Dtype>
	void CaffeWrapper<Dtype>::reshape(const int width, const int height)
	{
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_, height, width);
		net_.Reshape();
	}

	template<typename Dtype>
	cv::Mat CaffeWrapper<Dtype>::reshapeImage(const cv::Mat & image)
	{
		cv::Mat reshapedImage;
		cv::resize(image, reshapedImage, cv::Size(im_w, im_h), cv::INTER_CUBIC);
		return reshapedImage;
	}

	template<typename Dtype>
	void CaffeWrapper<Dtype>::forwardImage(const string& input_name, cv::Mat & im)
	{
		cv::Mat inputImage;
		if (im.cols != im_w || im.rows != im_h) {
			im = getImagePadResize(im, cv::Size(im_w, im_h));
		}
		im.convertTo(inputImage, CV_32F, 1.0 / 255, -0.5);
		CHECK(net_->has_blob(input_name)) << input_name << " does not exist. Please check";

		boost::shared_ptr<Blob<Dtype>> input_layer = net_->blob_by_name(input_name);
		input_layer->Reshape(1, num_channels_, im_h, im_w);
		net_->Reshape();
		int num_channels = input_layer->channels();
		Dtype* input_data = input_layer->mutable_cpu_data();
		int offset = im_w*im_h;
		vector<cv::Mat> input_channels(num_channels);
		for (auto i = 0; i < num_channels; ++i) {
			input_channels[i] = cv::Mat(im_h, im_w, CV_32FC1, input_data);
			input_data += offset;
		}
		cv::split(inputImage,input_channels);
		net_->Forward();
	}

	template<typename Dtype>
	void CaffeWrapper<Dtype>::forwardBlob(const string& input_name, CaffeBlob<Dtype>* net_input)
	{
		CHECK(net_->has_blob(input_name)) << input_name << " does not exist. Please check";
		shared_ptr<Blob<Dtype>> input_layer = net_->blob_by_name(input_name);
		Dtype* input_data = input_layer->mutable_cpu_data();
		for (auto i = 0; i < net_input->count; ++i) {
			input_data[i] = net_input->data[i];
		}
	}

	template<typename Dtype>
	void CaffeWrapper<Dtype>::getOutputBlob(const string & output_name, CaffeBlob<Dtype>* net_output)
	{
		CHECK(net_->has_blob(output_name)) << output_name << " does not exist. Please check";

		boost::shared_ptr<Blob<Dtype>> output_layer = net_->blob_by_name(output_name);
		Dtype* out_data = (Dtype*)output_layer->cpu_data();

		if (net_output->data_self) {
			for (auto i = 0; i < net_output->count; ++i) {
				net_output->data[i] = out_data[i];
			}
		}
		else {
			net_output->num = output_layer->num();
			net_output->channels = output_layer->channels();
			net_output->height = output_layer->height();
			net_output->width = output_layer->width();
			net_output->count = net_output->num*net_output->channels*net_output->height*net_output->width;
			net_output->capacity_count = net_output->count * sizeof(Dtype);
			net_output->data = out_data;
		}
	}

	template<typename Dtype>
	cv::Mat CaffeWrapper<Dtype>::getImagePadResize(cv::Mat & img, const cv::Size & baseSize)
	{
		//align the shorter edge to the baseSize
		int bw = baseSize.width;
		int bh = baseSize.height;
		int h = img.rows;
		int w = img.cols;

		cv::Mat reshape_(bh, bw, img.type());
		reshape_.setTo(0.0f);

		float scale_;
		float ratio1 = 1.0*w / h;
		float ratio2 = 1.0*bw / bh;
		Rect roi;
	
		if (ratio1 > ratio2) {	//align w
			scale_ = 1.0*bw / w;
			int nh = int(scale_*h);
			roi=Rect(0, 0, bw, nh);
		}
		else {
			scale_ = 1.0*bh / h;
			int nw = int(scale_*w);
			roi=Rect(0, 0, nw, bh);
		}
		cv::Mat roi_ = reshape_(roi);
		cv::resize(img, roi_, cv::Size(roi.width,roi.height), cv::INTER_CUBIC);
		scale_x = scale_;
		scale_y = scale_;
		return reshape_;
	}

	template<typename Dtype>
	cv::Mat CaffeWrapper<Dtype>::getImageResize(cv::Mat & img, const cv::Size & baseSize)
	{
		cv::Mat reshape_;
		cv::resize(img, reshape_, baseSize, cv::INTER_CUBIC);
		scale_x = 1.0*baseSize.width / img.cols;
		scale_y = 1.0*baseSize.height / img.rows;
		return reshape_;
	}

}
