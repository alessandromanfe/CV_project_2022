
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace cv;
using namespace std;

Mat markerMask, img;
Point prevPt(-1, -1);
int main(int argc, char** argv)
{
    Mat img0 = imread("rgb/04.jpg", 1), imgGray;
    ifstream in = ifstream("det/04.txt");
    vector<Rect> real;
    int x, y, widht, heigth;
    while (in >> x >> y >> widht >> heigth)
        real.push_back(Rect(x, y, widht, heigth));
    img0.copyTo(img);
    cvtColor(img, markerMask, COLOR_BGR2GRAY);
    cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
    Mat watersheddred(img.size(), CV_8UC3);
    for (int z = 0; z < real.size(); z++)
    {
        
        markerMask = Scalar::all(0);
        rectangle(markerMask, real[z], Scalar::all(255), 4, 8, 0);
        float ratio = 0.4;
        float centerx = real[z].x + (real[z].width / 2);
        float centery = real[z].y + (real[z].height / 2);
        //drawMarker(markerMask, Point(real[0].x + real[0].width / 2, real[0].y + real[0].height / 2), Scalar::all(255),0, 3, 8);

        line(markerMask, Point(centerx + ratio * real[z].width, centery), Point(centerx - ratio * real[z].width, centery), Scalar::all(255), 4, 8, 0);
        line(markerMask, Point(centerx, centery + ratio * real[z].height), Point(centerx, centery - ratio * real[z].height), Scalar::all(255), 4, 8, 0);

        int i, j, compCount = 0;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        Mat markers(markerMask.size(), CV_32S);
        markers = Scalar::all(0);
        int idx = 0;
        for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
            drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);

        vector<Vec3b> colorTab;
        for (i = 0; i < compCount; i++)
        {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);
            colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        watershed(img0, markers);
        Mat wshed(markers.size(), CV_8UC3);
        // paint the watershed image
        for (i = 0; i < markers.rows; i++)
            for (j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i, j);
                if (index == -1)
                    wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
                else if (index <= 0 || index > compCount)
                    wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                else
                    wshed.at<Vec3b>(i, j) = colorTab[index - 1];
            }
        wshed(real[z]).copyTo(watersheddred(real[z]));
    }
    watersheddred = watersheddred * 0.5 + imgGray * 0.5;
    imshow("watershed transform", watersheddred);
        
            waitKey(0);
    return 0;
}
