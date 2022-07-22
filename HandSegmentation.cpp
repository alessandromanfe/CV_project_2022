#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<vector>
using namespace std;
using namespace cv;

Mat src_img;
Mat img;
cv::Rect rect(0,0,0,0);
cv::Point p1(0,0);
cv::Point p2(0,0);
//using mouse
static bool clicked = false;
//create a function to fix boundries
void fix_boundries()
{
	if(rect.width >img.cols -rect.x)
	rect.width = img.cols - rect.x;
	if(rect.height > img.rows -rect.y)
	rect.height = img.rows -rect.y;
	if(rect.x < 0)
	rect.x = 0;
	if(rect.y < 0)
	rect.y = 0;
}

//create a function to draw a rectangle
void draw()
{
	img = src_img.clone();
	fix_boundries();
	cv::rectangle(img, rect, cv::Scalar(0,255,0),1,8,0);
	cv::imshow("Original image", img);
}

//create a function to control the area of drawn rectangle using mouse
void onMouse(int event, int x, int y, int flag, void* user_data)
{
	switch(event)
	{
		case EVENT_LBUTTONDOWN:
			clicked = true;
			p1.x = x;
			p1.y = y;
			p2.x = x;
			p2.y = y;
			break;
		case EVENT_LBUTTONUP:
			clicked = false;
			p2.x = x;
			p2.y = y;
			break;
		case EVENT_MOUSEMOVE:
			if(clicked)
			{
				p2.x = x;
				p2.y = y;
			}
			break;
		default:
			break;
			
			
	}
	if(p1.x > p2.x)
	{
		rect.x = p2.x;
		rect.width = p1.x - p2.x;
	}
	else
	{
		rect.x = p1.x;
		rect.width = p2.x - p1.x;
	}
	if(p1.y > p2.y)
	{
		rect.y = p2.y;
		rect.height = p1.y - p2.y;
	}
	else
	{
		rect.y = p1.y;
		rect.height = p2.y - p1.y;
	}
	draw();
}
int main(int argc, char *argv[])
{
    
    src_img= imread(argv[1]);
	namedWindow("Original image", WINDOW_NORMAL);
    setMouseCallback("Original image", onMouse, NULL);
    imshow("Original image", src_img);
    Mat result;
    Mat bgModel, fgModel;
    while(1){
        char c=waitKey(0);
        grabCut(src_img,result,rect,bgModel,fgModel,5,GC_INIT_WITH_RECT);
        compare(result,GC_PR_FGD,result,CMP_EQ);
        Mat foreground(src_img.size(),CV_8UC3,Scalar(255,255,255));
        src_img.copyTo(foreground,result);
        namedWindow("segmented img");
        imshow("segmented img",foreground);
        (char) waitKey(0);
        
    }
      
    return 0;
}



