#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>

using namespace std;
using namespace cv;

typedef struct {
    int p1;
    int p2;
    int minr;
    int maxr;
    int mindist;
    Mat* img;
    Mat* img_gray;
} hData;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

void trackbarCallback(int pos, void* data) {
    hData x = *((hData*)data);
    vector<Vec3f> circles;
    Mat hough = (*x.img).clone();
    HoughCircles((*x.img_gray), circles, HOUGH_GRADIENT, 1, x.img->rows/x.mindist, x.p1, x.p2, x.minr, x.maxr);
    for (size_t i = 0; i < (circles).size(); i++)
    {
        Vec3i c = (circles)[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle((hough), center, 1, Scalar(0, 100, 100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle((hough), center, radius, Scalar(255, 255, 0), 3, LINE_AA);
    }
    imshow("Hough", (hough));
}


Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch (src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

Mat tan_triggs_preprocessing(InputArray src,
    float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
    int sigma1 = 2) {

    // Convert to floating point:
    Mat X = src.getMat();
    X.convertTo(X, CV_32FC1);
    // Start preprocessing:
    Mat I;
    pow(X, gamma, I);
    // Calculate the DOG Image:
    {
        Mat gaussian0, gaussian1;
        // Kernel Size:
        int kernel_sz0 = (3 * sigma0);
        int kernel_sz1 = (3 * sigma1);
        // Make them odd for OpenCV:
        kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
        kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
        GaussianBlur(I, gaussian0, Size(kernel_sz0, kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
        GaussianBlur(I, gaussian1, Size(kernel_sz1, kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
        subtract(gaussian0, gaussian1, I);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(abs(I), alpha, tmp);
            meanI = mean(tmp).val[0];

        }
        I = I / pow(meanI, 1.0 / alpha);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(min(abs(I), tau), alpha, tmp);
            meanI = mean(tmp).val[0];
        }
        I = I / pow(meanI, 1.0 / alpha);
    }

    // Squash into the tanh:
    {
        Mat exp_x, exp_negx;
        exp(I / tau, exp_x);
        exp(-I / tau, exp_negx);
        divide(exp_x - exp_negx, exp_x + exp_negx, I);
        I = tau * I;
    }
    return I;
}

int main(int argc, char** argv) {

	char* lpath = "./eyes_direction_cv/left";
	char* cpath = "./eyes_direction_cv/center";
	char* rpath = "./eyes_direction_cv/right";

	vector<Mat> left;
	vector<Mat> center;
	vector<Mat> right;

	for (const auto& img : filesystem::directory_iterator(lpath)) {
		left.push_back(imread(img.path().string()));
	}
	for (const auto& img : filesystem::directory_iterator(cpath)) {
		left.push_back(imread(img.path().string()));
	}
	for (const auto& img : filesystem::directory_iterator(rpath)) {
		left.push_back(imread(img.path().string()));
	}

    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
        "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}");
    parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
        "You can use Haar or LBP features.\n\n");
    parser.printMessage();
    String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
    String eyes_cascade_name = samples::findFile(parser.get<String>("eyes_cascade"));
    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };

    namedWindow("Hough", WINDOW_AUTOSIZE);
    Mat left_gray;
    cvtColor(left[0], left_gray, COLOR_BGR2GRAY);
    hData d = { 255, 1, 1, 30, 30, &left[0], &left_gray};
    createTrackbar("p1", "Hough", &d.p1, 500, trackbarCallback, &d);
    createTrackbar("p2", "Hough", &d.p2, 100, trackbarCallback, &d);
    createTrackbar("minr", "Hough", &d.minr, 100, trackbarCallback, &d);
    createTrackbar("maxr", "Hough", &d.maxr, 100, trackbarCallback, &d);
    createTrackbar("mindist", "Hough", &d.mindist, 100, trackbarCallback, &d);

	for (Mat img : left) {
        Mat img_gray, hough = img.clone();
        Mat preproc = tan_triggs_preprocessing(img);
        Mat tt = norm_0_255(preproc);
        //hough = tt.clone();
        GaussianBlur(hough, hough, Size(3, 3), 0);
        cvtColor(hough, img_gray, COLOR_BGR2GRAY);
        //equalizeHist(img_gray, img_gray);
        //medianBlur(img_gray, img_gray, 5);

        d.img = &tt;
        d.img_gray = &img_gray;

        vector<Vec3f> circles;
        HoughCircles(img_gray, circles, HOUGH_GRADIENT, 1, img_gray.rows / d.mindist, d.p1, d.p2, d.minr, d.maxr);

        std::vector<Rect> faces;
        face_cascade.detectMultiScale(img_gray, faces, 1.02, 9);

        for (size_t i = 0; i < faces.size(); i++)
        {
            Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            ellipse(hough, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
            Mat faceROI = img_gray(faces[i]);
            //-- In each face, detect eyes
            std::vector<Rect> eyes;
            eyes_cascade.detectMultiScale(faceROI, eyes, 1.02, 9);
            for (size_t j = 0; j < eyes.size(); j++)
            {
                Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                circle(hough, eye_center, radius, Scalar(255, 0, 0), 4);
            }

            Mat face = img_gray(faces[i]);
            //HoughCircles(face, circles, HOUGH_GRADIENT, 1.5, img_gray.rows / 15, 300, 10, 1, 30);

            //threshold(face, face, 200, 255, THRESH_BINARY);
            imshow("face", face);
        }

        
        for (size_t i = 0; i < circles.size(); i++)
        {
            Vec3i c = circles[i];
            Point center = Point(c[0], c[1]);
            // circle center
            circle(hough, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
            // circle outline
            int radius = c[2];
            circle(hough, center, radius, Scalar(255, 255, 0), 3, LINE_AA);
        }
        /*
        Mat lap;
        Laplacian(img_gray, lap, CV_16S, 5);
        //threshold(lap, lap, 200, 33000, THRESH_BINARY);
        lap.convertTo(lap, CV_8U);
        //Mat z = Mat::zeros(lap.size(), CV_8U);
        //equalizeHist(lap,lap);
        imshow("lap", lap);

        vector<Vec3f> circles2;
        HoughCircles(lap, circles2, HOUGH_GRADIENT_ALT, 1, lap.rows / 30, 200, 0.8, 1, 50);
        for (size_t i = 0; i < circles2.size(); i++)
        {
            Vec3i c = circles2[i];
            Point center = Point(c[0], c[1]);
            // circle center
            circle(img, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
            // circle outline
            int radius = c[2];
            circle(img, center, radius, Scalar(0, 255, 255), 3, LINE_AA);
        }
        */

        


        //-- Show what you got
        imshow("Hough", hough);
        waitKey(0);

	}

	return 0;
}