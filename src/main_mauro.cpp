
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;

void faceDetection(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    // Detect faces     
    std::vector<Rect> eyes;
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces,1.0045,3,0,Size(100,100));
    int bestNumb=0;
    Mat bestfacepart;
    int pos=-1;
    //search best face aka the face with most eyes in it 
    for (int val = 0; val < faces.size(); val++)
    {
        
        Mat faceROI = frame_gray(faces[val]);
        eye_cascade.detectMultiScale(faceROI, eyes,1.02,9);
        if(eyes.size()>bestNumb)
        {
            bestfacepart=faceROI;
            bestNumb=eyes.size();
            pos=val;
        }
        
    }
    //if we do not find a face with eyes, we try to use the equalized image
    if(pos==-1)
    {
        Mat equal_img;
        equalizeHist(frame_gray, equal_img);
        for (int val = 0; val < faces.size(); val++)
        {
        
        Mat faceROI = equal_img(faces[val]);
        eye_cascade.detectMultiScale(faceROI, eyes,1.02,9);
        if(eyes.size()>bestNumb)
        {
            bestfacepart=faceROI;
            bestNumb=eyes.size();
            pos=val;
        }
        
        }
    }
    
    eyes.clear();    
    //printing best face
    Point center(faces[pos].x+ faces[pos].width / 2,faces[pos].y+ faces[pos].height / 2);
    ellipse(frame, center, Size(faces[pos].width / 2, faces[pos].height / 2), 0, 0, 360, Scalar(0, 0, 255), 6);
    
    
    eye_cascade.detectMultiScale(bestfacepart, eyes,1.02,9);



    int largest = 0, second_largest = 0, pos1=-1, pos2=-1;
    
    //search best eye
    for(int i = 0; i < eyes.size(); i++)
    {
        if(eyes[i].area()>largest)
        {
            largest=eyes[i].area();
            pos1=i;
        }
    }
    //search second best eye
    for(int i = 0; i < eyes.size(); i++)
    {
        if(eyes[i].area()>second_largest)
        {
            if( i != pos1)
            {
                second_largest=eyes[i].area();
                pos2=i;
            }
        }
            
    }
    //print EYES
    if(pos1!=-1)
    {
        Point centereye(faces[pos].x+eyes[pos1].x + eyes[pos1].width / 2,faces[pos].y+ eyes[pos1].y + eyes[pos1].height / 2);
        ellipse(frame, centereye, Size(eyes[pos1].width / 2, eyes[pos1].height / 2), 0, 0, 360, Scalar(0, 255, 0), 6);
    }
    if(pos2!=-1)
    {    
        Point centereye(faces[pos].x+eyes[pos2].x + eyes[pos2].width / 2,faces[pos].y+ eyes[pos2].y + eyes[pos2].height / 2);
        ellipse(frame, centereye, Size(eyes[pos2].width / 2, eyes[pos2].height / 2), 0, 0, 360, Scalar(0, 255, 0), 6);
    }
        
        
    //show img        
    imshow("Live Face Detection", frame);
}



int main(int argc, const char** argv)
{

    // Load the pre trained haar cascade classifier

    string faceClassifier = "haarcascade_frontalface_default.xml";
    string eyeClassifier = "haarcascade_eye_tree_eyeglasses.xml";
    //string faceClassifier = "haarcascade_frontalface_alt2.xml";

    if (!face_cascade.load(faceClassifier))
    {
        cout << "Could not load the classifier";
        return -1;
    };
    if (!eye_cascade.load(eyeClassifier))
    {
        cout << "Could not load the classifier";
        return -1;
    };


    vector<cv::String> fn;
    fn.push_back("imgs/left03.jpg");
    
    cv::glob("imgs/*.jpg", fn, false);
    vector<cv::String> fn1;
    cv::glob("imgs/*.jpeg", fn1, false);
    vector<cv::String> fn2;
    cv::glob("imgs/*.png", fn2, false);
    fn.insert( fn.end(), fn1.begin(), fn1.end() );
    fn.insert( fn.end(), fn2.begin(), fn2.end() );
    
    Mat frame;
    
    vector<cv::Mat> input;
    for(int j=0;j<fn.size();j++)
        input.push_back(cv::imread(fn[j]));
        
        
    for(int i=0;i<input.size();i++)
    {
        frame=input[i];
        // Apply the face detection with the haar cascade classifier
        faceDetection(frame);

        if (waitKey(1000) == 'q')
        {
            cout<<"xd"; // Terminate program if q pressed
        }
   }

    return 0;
}

