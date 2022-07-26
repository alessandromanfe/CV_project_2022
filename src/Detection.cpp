#include <fstream>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

//function for loading net
void load_net(cv::dnn::Net &net)
{
    cv::dnn::Net result = cv::dnn::readNet("new-yolov5m.onnx");
    result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net = result;
}

//constants threshold
const float input_size = 640.0;
const float NM_THRESH = 0.3;
const float CONF_THRESH = 0.19;

double IoU(const std::vector<cv::Rect>& boxes, const std::vector<cv::Rect>& real) {
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
/*     OLD IOU
bool doOverlap(cv::Rect a,cv::Rect b)
{
    cv::Point l1=cv::Point(a.x,a.y);
    cv::Point r1=cv::Point(a.x+a.width,a.y+a.height);
    cv::Point r2=cv::Point(b.x+b.width,b.y+b.height);
    cv::Point l2=cv::Point(b.x,b.y);
    if (l1.x == r1.x || l1.y == r1.y || r2.x == l2.x || l2.y == r2.y)
        return false;
    if (l1.x > r2.x || l2.x > r1.x)
        return false;
    if (r1.y < l2.y || r2.y < l1.y)
        return false;
    return true;
}

float IntOverUnion(cv::Rect a,cv::Rect b)
{
    float iou=0;
    if(doOverlap(a,b))
    {
        int intx,inty,intx2,inty2;
        intx=max(a.x,b.x);
        inty=max(a.y,b.y);
        intx2=min(a.x+a.width,b.x+b.width);
        inty2=min(a.y+a.height,b.y+b.height);
        cv::Rect intersect=cv::Rect(intx,inty,intx-intx2,inty-inty2);
        float areaint=intersect.area();
        float areunion=a.area()+b.area() - areaint;
        iou= areaint/areunion;
    }
    return iou;
}

float perfomEvaluation(vector<cv::Rect> boxes, vector<cv::Rect> real)
{
    float res=0;
    int num=max(boxes.size(),real.size());
    while(!boxes.empty() && !real.empty())
    {
        float max=0,temp=0;
        int idbox=0,idreal=0;
        for(int i=0;i<boxes.size();i++)
        {
            for(int j=0;j<real.size();j++)
            {
                temp=IntOverUnion(boxes[i],real[j]);
                if(temp>max)
                {
                    max=temp;
                    idbox=i;
                    idreal=j;
                }
            }
        }
        res+=max;
        boxes.erase(boxes.begin() + idbox);
        real.erase(real.begin() + idreal);
    }
    return res/num;
}
*/

cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

//detect of hands
void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<cv::Rect> &output) {
    cv::Mat blob;
    //trasform input image into imgage in square with black filled
    cv::Mat input_image = format_yolov5(image);
    
    //read with the net 
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(input_size, input_size), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    //taking scale factors for recenters the boxes
    float x_factor = input_image.cols / input_size;
    float y_factor = input_image.rows / input_size;
    
    float *data = (float *)outputs[0].data;
    
    const int rows = 25200;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    //for each detection (it does 25200 detections)
    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        //threshold the confidence
        if (confidence >= CONF_THRESH) {
                confidences.push_back(confidence);
                
                //find the rbox
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));

        }
        //move to next detection
        data += 6;

    }
    //apply non maxima suppresion to remove overlapping boxes that detect the same hand
    std::vector<int> nmaxsupp_res;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESH, NM_THRESH, nmaxsupp_res);
    //create output vector
    for (int i = 0; i < nmaxsupp_res.size(); i++) 
    {
        
        int num = nmaxsupp_res[i];
        output.push_back(boxes[num]);
    }
}

int main(int argc, char **argv)
{
    //load vector of input img
    vector<cv::String> fn;
    cv::glob("rgb/*.jpg", fn, false);
    cv::Mat frame;
    vector<cv::Mat> input;
    for(int j=0;j<fn.size();j++)
        input.push_back(cv::imread(fn[j]));

    //load real bound
    vector<cv::Rect> realBound;
    std::ifstream infile;

    vector<cv::String> tn;
    cv::glob("det/*.txt", tn, false);
        
    //load net
    cv::dnn::Net net;
    load_net(net);
    
    //start 
    float sum=0;
    //exectute for each image found
    for(int j=0;j<input.size();j++)
    {
        realBound.clear();
        //get img j
        frame=input[j];
        //read txt  and create rect for each bound box 
        infile=ifstream(tn[j]);
        int in_x ,in_y ,in_w ,in_h;
        while (infile >> in_x >> in_y >> in_w >> in_h)
        {
            realBound.push_back(cv::Rect(in_x, in_y, in_w, in_h));
        }
        //detect our box
        vector<cv::Rect> output;
        detect(frame, net, output);
        for (int i = 0; i < output.size(); ++i)
        {
            //take each box
            cv::Rect box = output[i];
            //find a random color
            const cv::Scalar color = cv::Scalar(rand()%255, rand()%255, rand()%255);
            //create rect and text
            cv::rectangle(frame, box, color, 1);
            cv::putText(frame, "Hand", cv::Point(box.x, box.y-1), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
            //show img
            //cv::namedWindow(fn[j],cv::WINDOW_NORMAL);
            //cv::imshow(fn[j], frame);
        }
        
        //calculation of IOU Metric
        float per=IoU(output,realBound);
        //cout<<per<<endl;
        
        //cv::waitKey(0);
    }
    return 0;
}
