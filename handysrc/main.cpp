#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <filesystem>
#include <fstream>

#include "BackgroundRemover.h"
#include "SkinDetector.h"
#include "FaceDetector.h"
#include "FingerCount.h"

using namespace cv;
using namespace std;

int pixelAccuracy(const cv::Mat& x, const cv::Mat& y, std::vector<double>& v) {
	if (x.channels() != 1 || y.channels() != 1) return -1;			//checks channels
	int rows = x.rows;
	int cols = x.cols;
	if (y.rows != rows || y.cols != cols) return -1;				//checks dimensions
	cv::Mat inter_mat, union_mat;
	bitwise_and(x, y, inter_mat);
	bitwise_or(x, y, union_mat);
	int hand_count = countNonZero(inter_mat);						//number of hand pixels correctly classified
	int hand_tot = countNonZero(x);									//number of pixel classified as hand
	double hand_accuracy = hand_count * 1.0 / hand_tot;
	//Here we consider intersection of negative masks which is negation of union
	int non_hand_count = rows * cols - countNonZero(union_mat);		//number of non-hand pixels correctly classified
	int non_hand_tot = rows * cols - countNonZero(x);				//number of pixel classified as non-hand
	double non_hand_accuracy = non_hand_count * 1.0 / non_hand_tot;
	v = std::vector<double>{ hand_accuracy,non_hand_accuracy };
	return 1;
}

void clusterSegment(const cv::Mat& src, cv::Mat& dst, int k, int attempts, cv::Mat& indices, cv::Mat& centers) {
	cv::Mat img_blur, data, labels, clusters;
	cv::GaussianBlur(src, img_blur, cv::Size(3, 3), 50);
	img_blur.convertTo(data, CV_32F);
	data = data.reshape(1, data.total());
	cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
	cv::kmeans(data, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);
	centers = centers.reshape(3, centers.rows);
	data = data.reshape(3, data.rows);
	cv::Vec3f* p = data.ptr<cv::Vec3f>();
	for (int i = 0; i < data.rows; i++) {
		int c_index = labels.at<int>(i);
		p[i] = centers.at<cv::Vec3f>(c_index);
	}
	clusters = data.reshape(3, src.rows);
	clusters.convertTo(clusters, CV_8U);
	dst = clusters.clone();
	labels = labels.reshape(0, src.rows);
	indices = labels.clone();
}

int main(int, char**) {
	/*
	VideoCapture videoCapture(0);
	videoCapture.set(CAP_PROP_SETTINGS, 1);

	if (!videoCapture.isOpened()) {
		cout << "Can't find camera!" << endl;
		return -1;
	}

	Mat frame, frameOut, handMask, foreground, fingerCountDebug;

	BackgroundRemover backgroundRemover;
	SkinDetector skinDetector;
	FaceDetector faceDetector;
	FingerCount fingerCount;

	while (true) {
		videoCapture >> frame;
		frameOut = frame.clone();

		skinDetector.drawSkinColorSampler(frameOut);

		foreground = backgroundRemover.getForeground(frame);
		
		faceDetector.removeFaces(frame, foreground);
		handMask = skinDetector.getSkinMask(foreground);
		fingerCountDebug = fingerCount.findFingersCount(handMask, frameOut);

		imshow("output", frameOut);
		imshow("foreground", foreground);
		imshow("handMask", handMask);
		imshow("handDetection", fingerCountDebug);
		
		int key = waitKey(1);

		if (key == 27) // esc
			break;
		else if (key == 98) // b
			backgroundRemover.calibrate(frame);
		else if (key == 115) // s
			skinDetector.calibrate(frame);
	}
	*/

	char* imgpath = "./Dataset/rgb";
	char* detpath = "./Dataset/det";
	char* maskpath = "./Dataset/mask";

	vector<Mat> images;
	vector<Mat> masks;
	vector<vector<Rect>> boxes;

	Mat frame, frameOut, handMask, foreground, fingerCountDebug;

	BackgroundRemover backgroundRemover;
	SkinDetector skinDetector;
	FaceDetector faceDetector;
	FingerCount fingerCount;

	for (const auto& img : filesystem::directory_iterator(imgpath)) {
		images.push_back(imread(img.path().string()));
	}
	for (const auto& img : filesystem::directory_iterator(maskpath)) {
		masks.push_back(imread(img.path().string(), IMREAD_GRAYSCALE));
	}
	for (const auto& item : filesystem::directory_iterator(detpath)) {
		ifstream detfile;
		detfile.open(item.path().string());
		if (!detfile) {
			cout << "errore lettura files" << endl;
			return -1;
		}
		vector<Rect> imgboxes;
		int x, y, w, h;
		while (detfile.good()) {
			detfile >> x >> y >> w >> h;
			imgboxes.push_back(Rect(x, y, w, h));
		}
		boxes.push_back(imgboxes);
		detfile.close();
	}

	vector<vector<Mat>> boxedimg;
	for (int i = 0; i < images.size(); i++) {
		vector<Mat> detboxes;
		for (Rect r : boxes[i]) {
			detboxes.push_back(images[i](r));
		}
		boxedimg.push_back(detboxes);
	}

	for (int i = 0; i < images.size(); i++) {
		Mat img = images[i], genmask = Mat::zeros(img.size(), CV_8U);
		vector<Rect> boxs = boxes[i];
		for (Rect r : boxs) {
			Mat back;
			img.copyTo(back);
			back(r) = 0;

			
			int size = 20;
			Rect r1(r.x + r.width / 2 - size * 1.5, r.y + r.height / 2 - size / 2, size, size);
			Rect r2(r.x + r.width / 2 + size * 0.5, r.y + r.height / 2 - size / 2, size, size);
			/*
			int size = r.area() / 1000;
			if (size > 20) size = 20;
			if (size < 4) size = 4;
			/*
			Mat centers, labels, clust;
			clusterSegment(img(r), clust, 4, 3, labels, centers);
			Vec2b c1 = centers.at<Vec2b>(0, 0);
			Vec2b c2 = centers.at<Vec2b>(0, 1);
			Rect r1(c1[0] + r.x - size * 0.5, c1[1] + r.y - size*0.5, size, size);
			Rect r2(c2[0] + r.x - size * 0.5, c2[1] + r.y - size * 0.5, size, size);
			*/

			backgroundRemover.calibrate(back);
			skinDetector.drawSkinColorSampler(img, r1, r2);
			skinDetector.calibrate(img);

			foreground = backgroundRemover.getForeground(img);
			faceDetector.removeFaces(img, foreground);
			handMask = skinDetector.getSkinMask(foreground);
			fingerCountDebug = fingerCount.findFingersCount(handMask, frameOut);
			
			bitwise_or(handMask, genmask, genmask);
		}
		vector<double> acc;
		if (pixelAccuracy(genmask, masks[i], acc)) {
			cout << "Pixel acc " << i << ",  hand: " << acc[0] << " , non-hand: " << acc[1] << endl;
		}
		else cout << "Accuracy unavailable";
		Mat out, remask = Mat::zeros(img.size(), CV_8UC3), red = Mat::ones(img.size(), CV_8UC3)*(0,0,255);
		red.copyTo(remask, genmask);
		addWeighted(img, 0.7, remask, 0.3, 0, out);
		imshow("out", out);
		waitKey(0);
		
	}

	return 0;
}