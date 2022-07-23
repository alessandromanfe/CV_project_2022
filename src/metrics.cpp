#include "metrics.h"

int metrics::pixelAccuracy(const cv::Mat& x, const cv::Mat& y, std::vector<double>& v) {
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