#include <iostream>  
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "intensity_transformations/IntensityTrans.h"

using namespace std;

int main(int argc, char **argv)
{
    // Mat image_src = imread("/home/liubin/workspace/DigitalImageProcess_cpp/images/lena_gray_256.tif");
    Mat image_src = imread("/home/liubin/workspace/DigitalImageProcess_cpp/images/lena_color_256.tif");
    Mat gray_image_src;
    cvtColor(image_src, gray_image_src, COLOR_BGR2GRAY);
    IntensityTrans intensity_trans(gray_image_src);
    // intensity_trans.getHist();
    // intensity_trans.getReverseTrans();
    // intensity_trans.getLineTrans();
    // intensity_trans.getLogTrans();
    // intensity_trans.getGamaTrans();
    // intensity_trans.getThresholdTrans();
    // intensity_trans.getSegLineTrans();
    // intensity_trans.getIntensitySlicingTrans();
    // intensity_trans.getBitSlicingTrans();
    intensity_trans.getHistEqualize();
    waitKey(0);
    return 0;
}