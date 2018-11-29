#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

class IntensityTrans
{
private:
    Mat image;

public:
    IntensityTrans(Mat &input_img);
    ~IntensityTrans();

public:
    void getHist();
    void getHistEqualize();
    void getReverseTrans();
    void getLineTrans();
    void getLogTrans();
    void getGamaTrans();
    void getThresholdTrans();
    void getSegLineTrans();
    void getIntensitySlicingTrans();
    void getBitSlicingTrans();
};