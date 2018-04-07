#include "stdafx.h"
#include "handengine.h"

namespace op {
	HandEngine::HandEngine()
	{
		std::string prototxt_file = "I:/Develop/caffe/openpose_caffecpp/openpose/openpose/pose_deploy.prototxt";
		std::string caffemodel_file = "I:/Develop/caffe/openpose_caffecpp/openpose/openpose/pose_iter_102000.caffemodel";
		wrapper.reset(new CaffeWrapper<float>(prototxt_file, caffemodel_file));
		input_blob.reset(new CaffeBlob<float>(1, hand_num_parts + 1, wrapper->im_h, wrapper->im_w));
		peak_blob.reset(new CaffeBlob<float>(1, hand_num_parts, hand_max_people, 3));
	}


	HandEngine::~HandEngine()
	{
	}

	cv::Rect2f HandEngine::getHandRectangleByPose(vector<float>& armKeypoints, float thresh)
	{
		float wristScore = armKeypoints[2];
		float elbowScore = armKeypoints[5];
		float shoulderScore = armKeypoints[8];
		//cout << wristScore << "," << elbowScore << "," << shoulderScore << endl;
		if (wristScore < thresh || elbowScore < thresh || shoulderScore < thresh)
			return cv::Rect2f(0,0,0,0);
		cv::Rect2f hand;
		hand.x = armKeypoints[0] + 0.3f*(armKeypoints[0] - armKeypoints[3]);
		hand.y = armKeypoints[1] + 0.3f*(armKeypoints[1] - armKeypoints[4]);
		float distWE = sqrt((armKeypoints[0] - armKeypoints[3])*(armKeypoints[0] - armKeypoints[3]) + (armKeypoints[1] - armKeypoints[4])*(armKeypoints[1] - armKeypoints[4]));
		float distES = sqrt((armKeypoints[3] - armKeypoints[6])*(armKeypoints[3] - armKeypoints[6]) + (armKeypoints[4] - armKeypoints[7])*(armKeypoints[4] - armKeypoints[7]));
		hand.width = 1.2f*max(distWE, 0.9f*distES);
		hand.height = hand.width;
		hand.x -= hand.width*0.5f;
		hand.y -= hand.height*0.5f;
		hand.x = max(0.0f, hand.x);
		hand.y = max(0.0f, hand.y);
		return hand;
	}

	void HandEngine::getHandRectangles(const vector<float>& poseKeypoints, const vector<int>& shape)
	{
		int numPeople = shape[0];
		int lWristIndex = 7;
		int lElbowIndex = 6;
		int lShoulderIndex = 5;
		int rWristIndex = 4;
		int rElbowIndex = 3;
		int rShoulderIndex = 2;
		int offset = shape[1] * shape[2];

		vector<array<cv::Rect2f, 2>>().swap(handRectangles);
		for (auto person = 0; person < numPeople; ++person) {
			int start = person*offset;
			vector<float> lArmPose = {
				poseKeypoints[start + lWristIndex * shape[2]],poseKeypoints[start + lWristIndex * shape[2] + 1],poseKeypoints[start + lWristIndex * shape[2] + 2],
				poseKeypoints[start + lElbowIndex * shape[2]], poseKeypoints[start + lElbowIndex * shape[2] + 1],poseKeypoints[start + lElbowIndex * shape[2] + 2],
				poseKeypoints[start + lShoulderIndex * shape[2]],poseKeypoints[start + lShoulderIndex * shape[2] + 1],poseKeypoints[start + lShoulderIndex * shape[2] + 2]
			};
			vector<float> rArmPose = {
				poseKeypoints[start + rWristIndex * shape[2]],poseKeypoints[start + rWristIndex * shape[2] + 1],poseKeypoints[start + rWristIndex * shape[2] + 2],
				poseKeypoints[start + rElbowIndex * shape[2]],poseKeypoints[start + rElbowIndex * shape[2] + 1],poseKeypoints[start + rElbowIndex * shape[2] + 2],
				poseKeypoints[start + rShoulderIndex * shape[2]],poseKeypoints[start + rShoulderIndex * shape[2] + 1],poseKeypoints[start + rShoulderIndex * shape[2] + 2]
			};

			array<cv::Rect2f, 2> handsPerson = {
				getHandRectangleByPose(lArmPose),
				getHandRectangleByPose(rArmPose)
			};
			handRectangles.push_back(handsPerson);
		}
	}

	cv::Mat HandEngine::cropFrame(cv::Rect & handRect, cv::Mat & im, const bool mirror)
	{
		cv::Size baseSize(wrapper->im_w, wrapper->im_h);
		float ratio_w = 1.0* handRect.width / baseSize.width;
		float ratio_h = 1.0* handRect.height / baseSize.height;
		float ratio = ratio_w < ratio_h ? ratio_w : ratio_h;
		cv::Mat affineMat=cv::Mat::eye(2,3,CV_32F);
		cv::Mat affined;
		handRect.width = min(handRect.width, im.cols - handRect.x);
		handRect.height = min(handRect.height, im.rows - handRect.y);
		affineMat.at<float>(0, 0) = mirror?-ratio:ratio;
		affineMat.at<float>(1, 1) = ratio;
		affineMat.at<float>(0, 2) = mirror ? handRect.width + handRect.x : handRect.x;
		affineMat.at<float>(1, 2) = handRect.y;
		affineMatrix = affineMat.clone();
		cv::warpAffine(im, affined, affineMat, baseSize,
			CV_INTER_LINEAR | CV_WARP_INVERSE_MAP, cv::BORDER_CONSTANT, cv::Scalar{ 0,0,0 });
		//cv::imshow("rect", im(handRect));
		//cv::imshow("affined", affined);
		return affined;
	}

	void HandEngine::findPeaks(CaffeBlob<float>* heatmaps, CaffeBlob<float>* out, float thresh)
	{
		float* heatmap_ptr = heatmaps->data;
		float* out_ptr = out->data;
		int heatmap_offset = heatmaps->width*heatmaps->height;
		int out_offset = out->width*out->height;
		
		for (auto n = 0; n < heatmaps->num; ++n) {
			for (auto c = 0; c < heatmaps->channels - 1; ++c) {
				int num_peaks = 0;
				for (auto y = 1; y < heatmaps->height-1 && num_peaks < hand_max_people; ++y) {
					for (auto x = 1; x < heatmaps->width-1 && num_peaks < hand_max_people; ++x) {
						int center = y*heatmaps->width + x;
						float val = heatmap_ptr[center];
						if (val > thresh) {
							vector<std::tuple<int,int,float>> grids = {
								std::make_tuple(y-1,x-1,heatmap_ptr[center-heatmaps->width-1]),
								std::make_tuple(y-1,x,heatmap_ptr[center-heatmaps->width]),
								std::make_tuple(y-1,x+1,heatmap_ptr[center-heatmaps->width + 1]),
								std::make_tuple(y,x-1,heatmap_ptr[center - 1]),
								std::make_tuple(y,x+1,heatmap_ptr[center + 1]),
								std::make_tuple(y+1,x-1,heatmap_ptr[center+heatmaps->width  - 1]),
								std::make_tuple(y+1,x,heatmap_ptr[center+heatmaps->width]),
								std::make_tuple(y+1,x+1,heatmap_ptr[center+heatmaps->width + 1])
							};
							bool ok = true;
							for (auto k = 0; k < 8; ++k) {
								ok &= (val > std::get<2>(grids[k]));
							}
							if (ok) {
								float valAcc = 0;
								float xAcc = 0;
								float yAcc = 0;
								for (auto k = 0; k < 8; ++k) {
									yAcc += std::get<2>(grids[k])*std::get<0>(grids[k]);
									xAcc += std::get<2>(grids[k])*std::get<1>(grids[k]);
									valAcc += std::get<2>(grids[k]);
								}
								yAcc /= valAcc;
								xAcc /= valAcc;
								valAcc = val;
								out_ptr[(num_peaks+1) * 3] = xAcc;
								out_ptr[(num_peaks+1) * 3 + 1] = yAcc;
								out_ptr[(num_peaks+1)* 3 + 2] = valAcc;
								++num_peaks;
							}
						}
					}
				}
				out_ptr[0] = num_peaks;
				heatmap_ptr += heatmap_offset;
				out_ptr += out_offset;
			}
		}
	}

	void HandEngine::connectKeypoints(float* handKeypoints)
	{
		float* peak_ptr = peak_blob->data;
		int peak_offset = peak_blob->width*peak_blob->height;
		int keypoint_offset = hand_num_parts * 3;
		for (auto n = 0; n < 1; ++n) {	
			for (auto c = 0; c < peak_blob->channels; ++c) {
				if (peak_ptr[0] >0.5) {
					auto x = peak_ptr[3];
					auto y = peak_ptr[4];
					auto score = peak_ptr[5];
					handKeypoints[0] = x*affineMatrix.at<float>(0, 0) + y*affineMatrix.at<float>(0, 1) + affineMatrix.at<float>(0, 2);
					handKeypoints[1] = x*affineMatrix.at<float>(1, 0) + y*affineMatrix.at<float>(1, 1) + affineMatrix.at<float>(1, 2);
					handKeypoints[2] = score;
				}
				peak_ptr += peak_offset;
				handKeypoints += 3;
			}
		}
	}

	void HandEngine::detectHandKeypoints(cv::Mat hand_im)
	{
		wrapper->forwardImage("image", hand_im);
		CaffeBlob<float> blob;
		wrapper->getOutputBlob("net_output", &blob);
		cv::Mat heatmap = blob.ToImage();

		vector<cv::Mat> heatmaps(blob.channels);
		cv::Mat heatmapOrigin;

		cv::split(heatmap, heatmaps);

		float* input_data = input_blob->data;
		int offset = wrapper->im_h*wrapper->im_w;

		for (auto i = 0; i < blob.channels; ++i) {
			cv::Mat heatmapOrigin(wrapper->im_h, wrapper->im_w, CV_32F, input_data);
			cv::resize(heatmaps[i], heatmapOrigin, cv::Size(wrapper->im_w, wrapper->im_h), cv::INTER_CUBIC);
			//cv::normalize(heatmapOrigin, heatmapOrigin, 0, 1, NORM_MINMAX);
			input_data += offset;
		}
		//cv::Mat im_heatmap(wrapper->im_h, wrapper->im_w, CV_32F, input_blob->data+10*offset);
		//cv::normalize(im_heatmap, im_heatmap, 0, 1, NORM_MINMAX);
		//cv::imshow("heatmap",im_heatmap);
		findPeaks(input_blob.get(), peak_blob.get());
	}

	void HandEngine::renderHandKeypointsCpu(Mat & frame, CaffeBlob<float>* blobKeypoints, const float renderThreshold, float scale_x, float scale_y)
	{
		float* kp_ptr = blobKeypoints->data;
		float* pair_ptr = blobKeypoints->data;
		for (auto person = 0; person < blobKeypoints->num; ++person) {
			for (auto hand = 0; hand < blobKeypoints->channels; ++hand) {
				for (auto part = 0; part < blobKeypoints->height; ++part) {
					auto colorIndex = part * 3;
					int x = (int)(kp_ptr[0]*scale_x);
					int y = (int)(kp_ptr[1]*scale_y);
					float score = kp_ptr[2];
					if (score > renderThreshold) {
						cv::circle(frame, cv::Point(x, y), 2,
							cv::Scalar(HAND_COLORS_RENDER[colorIndex], HAND_COLORS_RENDER[colorIndex + 1], HAND_COLORS_RENDER[colorIndex + 2]),-1);
					}
					kp_ptr += 3;
				}
				for (auto pair = 0; pair < blobKeypoints->height; ++pair) {
					auto pairIndex = pair * 2;
					int startIndex = HAND_PAIRS_RENDER[pairIndex];
					int endIndex = HAND_PAIRS_RENDER[pairIndex + 1];
					int startX = (int)(pair_ptr[startIndex * 3]*scale_x);
					int startY = (int)(pair_ptr[startIndex * 3 + 1]*scale_y);
					float startScore = pair_ptr[startIndex *3 + 2];
					int endX= (int)(pair_ptr[endIndex * 3]*scale_x);
					int endY= (int)(pair_ptr[endIndex * 3+1]*scale_y);
					float endScore = pair_ptr[endIndex * 3 + 2];
					if (startScore > renderThreshold && endScore > renderThreshold) {
						cv::line(frame, cv::Point(startX, startY), cv::Point(endX, endY),
							cv::Scalar(HAND_COLORS_RENDER[endIndex], HAND_COLORS_RENDER[endIndex + 1], HAND_COLORS_RENDER[endIndex + 2]), 2);
					}
				}
				pair_ptr += blobKeypoints->height*blobKeypoints->width;
			}
		}
	}

	cv::Mat HandEngine::handKeypointsFromImage(cv::Mat & im, const vector<float>& poseKeypoints, const vector<int>& shape)
	{
		int numPeople=shape[0];
		getHandRectangles(poseKeypoints, shape);

		CaffeBlob<float> handKeypoints(numPeople,2,hand_num_parts,3);
		float* keypoint_ptr = handKeypoints.data;
		int offset = hand_num_parts * 3;	//63
		for (auto person = 0; person<numPeople; ++person) {
			for (auto hand = 0; hand < 2; ++hand) {
				const bool mirror = (hand==0);
				cv::Rect2f& hand_rectf = handRectangles[person][hand];
				cv::Rect hand_rect((int)hand_rectf.x, (int)hand_rectf.y, (int)hand_rectf.width, (int)hand_rectf.height);
				if (hand_rect.width > 0 && hand_rect.height > 0) {
					cv::Mat hand_im = cropFrame(hand_rect, im, mirror);
					detectHandKeypoints(hand_im);
					connectKeypoints(keypoint_ptr);
				}
				keypoint_ptr += offset;
			}
		}
		cv::Mat canvas = im.clone();
		renderHandKeypointsCpu(canvas, &handKeypoints, 0.05);
		return canvas;
	}
}
