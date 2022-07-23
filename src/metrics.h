#ifndef metrics
#define metrics

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace metric {
	/**
	 * Computes pixel accuracy for both hand and non-hand segmentation
	 * @param x generated mask to be evaluated
	 * @param y ground truth mask
	 * @param v vector of double of accuracies (hand accuracy, non-hand accuracy)
	 * @return 1 if the two matrices are same size and one channel, -1 otherwise
	 */
	int pixelAccuracy(const cv::Mat& x, const cv::Mat& y, std::vector<double>& v);

	/**
	 * Computes Intersection over Union for the given set of boxes
	 * @param boxes generated boxes to be evaluated
	 * @param real ground truth boxes
	 * @return IoU metric [0,1] as a double
	 */
	double IoU(const std::vector<cv::Rect>& boxes, const std::vector<cv::Rect>& real);
}

#endif
