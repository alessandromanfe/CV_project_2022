#include "metrics.h"

int metric::pixelAccuracy(const cv::Mat& x, const cv::Mat& y, std::vector<double>& v) {
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

double metric::IoU(const std::vector<cv::Rect>& boxes, const std::vector<cv::Rect>& real) {
	std::vector<cv::Rect> allbox;
	for (cv::Rect r : boxes) allbox.push_back(r);
	for (cv::Rect r : real) allbox.push_back(r);
	int min_x=INT32_MAX, min_y=INT32_MAX, max_x=0, max_y=0;
	for (cv::Rect r : allbox) {										//finds min dimensions of window
		if (r.tl().x < min_x) min_x = r.tl().x;
		if (r.tl().y < min_y) min_y = r.tl().y;
		if (r.br().x > max_x) max_x = r.br().x;
		if (r.br().y > max_y) max_y = r.br().y;
	}
	cv::Mat x = cv::Mat::zeros(cv::Size(max_x-min_x,max_y-min_y),CV_8U);
	cv::Mat y = cv::Mat::zeros(cv::Size(max_x - min_x, max_y - min_y), CV_8U);
	for (cv::Rect r : boxes) {										//fills all generated boxes
		cv::rectangle(x, cv::Point(r.tl().x - min_x, r.tl().y - min_y),
			cv::Point(r.br().x - min_x, r.br().y - min_y), 255, cv::FILLED);
	}																
	for (cv::Rect r : real) {										//fills all real boxes
		cv::rectangle(y, cv::Point(r.tl().x - min_x, r.tl().y - min_y),
			cv::Point(r.br().x - min_x, r.br().y - min_y), 255, cv::FILLED);
	}
	//Computes union and intersection of obtained images
	cv::Mat inter_mat, union_mat;
	bitwise_and(x, y, inter_mat);								
	bitwise_or(x, y, union_mat);
	int inter_count = countNonZero(inter_mat);					
	int union_count = countNonZero(union_mat);									
	return inter_count * 1.0 / union_count;
}