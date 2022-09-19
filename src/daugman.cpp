#include "daugman.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace std;
using namespace cv;

vector<double> dm::daugman(Mat img, Point p, int start_r, int end_r, int step) {

    vector<double> values;                          
    Mat mask = Mat::zeros(img.size(), CV_8U);

    //find sum of pixel on the circumference for each radius value
    for (int r = start_r;
        //keep the circle inside the image
        r <= end_r && p.x + r < img.cols && p.x - r >= 0 && p.y + r < img.rows && p.y - r >= 0;
        r += step) {

        //sum values according to the mask
        circle(mask, p, r, 255, 1);
        Mat diff;
        bitwise_and(img, mask, diff);
        double val = cv::sum(diff)[0];

        //normalize the sum
        values.push_back(val / (2 * M_PI * r));

        //reset the mask
        mask.setTo(0);
    }
    //if not enough values retrun an empty vector
    if (values.size() < 2) return vector<double>{ 0, 0 };

    //gaussian blurred vector of differences
    vector<double> delta;
    for (int i = 1; i < values.size(); i++) delta.push_back(values[i - 1] - values[i]);
    GaussianBlur(delta, delta, Size(1, 5), 0);
    for (int i = 0; i < delta.size(); i++) delta[i] = abs(delta[i]);

    //find argmax and radius
    auto it = max_element(delta.begin(), delta.end());
    int index = it - delta.begin();
    double radius = start_r + index * step;

    //find average of pixel values inside detected iris
    Mat a_diff, a_mask = Mat::zeros(img.size(), CV_8U);
    circle(a_mask, p, radius, 255, FILLED);
    //Mat kernel = getGaussianKernel(2*radius, 0, CV_32F);
    bitwise_and(img, a_mask, a_diff);
    //a_diff.mul(kernel);
    double sum = cv::sum(a_diff)[0];
    //parameter inversely proportional to pixel intensities to weight delta
    double alpha = 1 - sum / (pow(radius, 2) * M_PI * 255);


    vector<double> data;
    int deg = 30;
    double area = pow(radius, 2) * M_PI * deg / 360;
    for (int i = deg; i <= 360; i += deg) {
        ellipse(mask, p, Size(radius, radius), i, 0, deg, 255, FILLED, LINE_AA);
        bitwise_and(img, mask, mask);
        double val = cv::sum(mask)[0];
        data.push_back(val / area);
        mask.setTo(0);
    }
    double mean = 0;
    double var = 0;
    for (double d : data) mean += d;
    mean = mean / data.size();
    for (double d : data) var += pow((mean - d),2);
    var = var / data.size();

    //double alpha = 1 - mean / 255;
    double beta = min(1.0, 1 / log10(var));

    

    return vector<double>{ delta[index] * pow(alpha, 2) * pow(beta,1/3), radius };
}


vector<int> dm::findIris(Mat img, int d_start, int d_end, int d_step, int p_step) {

    if (img.cols != img.rows) cout << "Image should be a square for better performance";

    //create a grid of points coordinate 
    vector<Point> points;
    for (int i = img.cols * 1 / 5; i <= img.cols * 4 / 5; i += p_step) {
        for (int j = img.rows * 1 / 3; j <= img.rows * 2 / 3; j += p_step) {
            points.push_back(Point(i, j));
        }
    }

    //apply daugman to each point and find best result (max delta)
    vector<int> coordVal;
    int max = 0;
    int max_index = 0;
    for (int i = 0; i < points.size(); i++) {
        vector<double> val = dm::daugman(img, points[i], d_start, d_end, d_step);
        if (val[0] > max) {
            max = val[0];
            max_index = i;

            //return coordinates of center and radius
            coordVal = vector<int>{ points[i].x, points[i].y, (int)val[1] };
        }
    }
    return coordVal;
}


vector<int> dm::printIris(Mat src, Mat* dst, Rect r) {

    cout << r.width << " ,  " << r.height << endl;
    Mat eye = src(r);
    *dst = src.clone();
    Mat eye_gray, eye_up, eye_gauss;
    cvtColor(eye, eye_gray, COLOR_BGR2GRAY);
    double scale = 1;
    if (eye_gray.rows <= 150) {
        scale = eye_gray.rows / 150.f;
        resize(eye_gray, eye_up, Size(150, 150), 0, 0, INTER_CUBIC);
    }
    else eye_up = eye_gray.clone();
    equalizeHist(eye_up, eye_up);
    GaussianBlur(eye_up, eye_up, Size(3, 3), 0);
    vector<int> coordVal = dm::findIris(eye_up, 20, 70, 3, 5);
    Point center(coordVal[0] * scale + r.x, coordVal[1] * scale + r.y);
    int radius = coordVal[2] * scale;
    circle(*dst, center, 1, Scalar(100, 100, 100), 3, LINE_AA);
    circle(*dst, center, radius, Scalar(255, 255, 0), 1, LINE_AA);

    Point center2(coordVal[0], coordVal[1]);
    int r2 = coordVal[2];

    Mat mask1 = Mat::zeros(eye_up.size(), CV_8U);
    Mat mask2 = Mat::zeros(eye_up.size(), CV_8U);
    rectangle(mask1, Rect(center2.x - 3 * r2, center2.y - r2, 2 * r2, 2 * r2), 255, FILLED, LINE_AA);
    rectangle(mask2, Rect(center2.x + r2, center2.y - r2, 2 * r2, 2 * r2), 255, FILLED, LINE_AA);
    bitwise_and(eye_up, mask1, mask1);
    bitwise_and(eye_up, mask2, mask2);
    double right = sum(mask1)[0] / (4.f * pow(r2, 2));
    double left = sum(mask2)[0] / (4.f * pow(r2, 2));
    double eyepos = left - right;
    if (eyepos > 50) putText(eye_up, "Left", Point(5, 20), HersheyFonts::FONT_HERSHEY_PLAIN, 1, 255, 1);
    else if (eyepos < -50) putText(eye_up, "Right", Point(5, 20), HersheyFonts::FONT_HERSHEY_PLAIN, 1, 255, 1);
    else putText(eye_up, "Center", Point(5, 20), HersheyFonts::FONT_HERSHEY_PLAIN, 1, 255, 1);

    circle(eye_up, center2, 1, Scalar(100, 100, 100), 3, LINE_AA);
    circle(eye_up, center2, r2, Scalar(255, 255, 0), 1, LINE_AA);
    imshow("upscaled image", eye_up);
    waitKey(0);

    return coordVal;
}


