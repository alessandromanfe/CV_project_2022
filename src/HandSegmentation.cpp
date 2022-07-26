#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Vec3b RandomColor(int value);  //Generate random color function

int main(int argc, char* argv[])
{
	Mat image = imread("2.jpg");    //Load RGB color image
	imshow("Source Image", image);

	//Grayscale, filtering, Canny edge detection
	Mat imageGray;
	cvtColor(image, imageGray, COLOR_BGR2GRAY);//Gray conversion
	GaussianBlur(imageGray, imageGray, Size(5, 5), 2);   //Gaussian filtering
	imshow("Gray Image", imageGray);
	Canny(imageGray, imageGray, 80, 150);
	imshow("Canny Image", imageGray);

	//Find profile
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);  //outline	
	Mat marks(image.size(), CV_32S);   //Opencv watershed second matrix parameter
	marks = Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for (; index >= 0; index = hierarchy[index][0], compCount++)
	{
		//Mark marks and number the contours of different areas, which is equivalent to setting water injection points. There are as many water injection points as there are contours
		drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
	}

	//Let's look at what's in the incoming matrix marks
	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	imshow("marksShow", marksShows);
	imshow("outline", imageContours);
	watershed(image, marks);

	//Let's take a look at what is in the matrix marks after the watershed algorithm
	Mat afterWatershed;
	convertScaleAbs(marks, afterWatershed);
	imshow("After Watershed", afterWatershed);

	//Color fill each area
	Mat PerspectiveImage = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < marks.rows; i++)
	{
		for (int j = 0; j < marks.cols; j++)
		{
			int index = marks.at<int>(i, j);
			if (marks.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else
			{
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	imshow("After ColorFill", PerspectiveImage);

	//The result of segmentation and color filling is fused with the original image
	Mat wshed;
	addWeighted(image, 0.4, PerspectiveImage, 0.6, 0, wshed);
	imshow("AddWeighted Image", wshed);

	waitKey();
}

Vec3b RandomColor(int value)//Generate random color function</span>
{
	value = value % 255;  //Generate random numbers from 0 to 255
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}

