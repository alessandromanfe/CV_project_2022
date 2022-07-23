#ifndef metrics
#define metrics

#include <opencv2/highgui.hpp>

namespace metrics {
	/**
	 * Computes pixel accuracy for both hand and non-hand segmentation
	 * @param x generated mask to be evaluated
	 * @param y ground truth mask
	 * @param v vector of double of accuracies (hand accuracy, non-hand accuracy)
	 * @return 1 if the two matrices are same size and one channel, -1 otherwise
	 */
	int pixelAccuracy(const cv::Mat& x, const cv::Mat& y, std::vector<double>& v);
}

#endif
