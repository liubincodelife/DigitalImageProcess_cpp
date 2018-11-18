#include "IntensityTrans.h"
#include <math.h>

IntensityTrans::IntensityTrans(Mat &input_img)
{
    image = input_img.clone();

	int nChannels = image.channels();
	int nRows = image.rows;
	int nCols = image.cols * nChannels;
	
	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;

	namedWindow("IntensityTransSrcImage", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTransSrcImage", image);
}

IntensityTrans::~IntensityTrans()
{


}

//注释是自己理解，若有错误，欢迎批评指正！
Mat getHistograph_cv(const Mat grayImage)
{
	//定义求直方图的通道数目，从0开始索引
	int channels[]={0};
	//定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数
	//如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目
	const int histSize[]={256};
	//每一维bin的变化范围
	float range[]={0,256};
 
	//所有bin的变化范围，个数跟channels应该跟channels一致
	const float* ranges[]={range};
 
	//定义直方图，这里求的是直方图数据
	Mat hist;
	//opencv中计算直方图的函数，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数
	calcHist(&grayImage,1,channels,Mat(),hist,1,histSize,ranges,true,false);
 	cout<<"cv::hist"<<hist<<endl;
	//找出直方图统计的个数的最大值，用来作为直方图纵坐标的高
	double maxValue=0;
	//找矩阵中最大最小值及对应索引的函数
	minMaxLoc(hist,0,&maxValue,0,0);
	cout<<"cv::max value = "<<maxValue<<endl;
	//最大值取整
	int rows=cvRound(maxValue);
	cout<<"cv::histrows = "<<rows<<endl;
	//定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)
	//因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像
	Mat histImage=Mat::zeros(rows,256,CV_8UC1);
 
	//直方图图像表示
	for(int i=0;i<256;i++)
	{
		//取每个bin的数目
		int temp=(int)(hist.at<float>(i,0));
		//如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色
		//如果图像上有该灰度值，则将该列对应个数的像素设为白色
		if(temp)
		{
			//由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点
			histImage.col(i).rowRange(Range(rows-temp,rows)) = 255; 
		}
	}
	//由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观
	Mat resizeImage;
	resize(histImage,resizeImage,Size(256,256));
	return resizeImage;
}

Mat getHistograph_cpp(const Mat grayimage)
{
	Mat histimg;
	int graycalc[256] = {0};
	int nChannels = grayimage.channels();
	int nRows = grayimage.rows;
	int nCols = grayimage.cols * nChannels;
	
	// uchar grayvalue = 0;
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			uchar grayvalue = grayimage.at<uchar>(i, j);
			graycalc[grayvalue]++;
		}
	}

	//找出直方图统计的个数的最大值，用来作为直方图纵坐标的高
	double maxValue=0;
	Mat graymat(256, 1, CV_16UC1, graycalc);
	cout<<graymat<<endl;
	//找矩阵中最大最小值及对应索引的函数
	minMaxLoc(graymat,0,&maxValue,0,0);
	cout<<"max value = "<<maxValue<<endl;
	//最大值取整
	int histrows=cvRound(maxValue);
	cout<<"histrows = "<<histrows<<endl;
	//定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)
	//因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像
	Mat histImage=Mat::zeros(histrows,256,CV_8UC1);
 
	//直方图图像表示
	for(int i=0;i<256;i++)
	{
		//取每个bin的数目
		int temp = graycalc[i];
		//如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色
		//如果图像上有该灰度值，则将该列对应个数的像素设为白色
		if(temp)
		{
			//由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点
			histImage.col(i).rowRange(Range(histrows-temp,histrows))=255; 
		}
	}
	//由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观
	Mat resizeImage;
	resize(histImage,resizeImage,Size(256,256));
	return resizeImage;
}

void IntensityTrans::getHist()
{
    Mat outimage_cv, outimage_cpp;
	
    outimage_cv = getHistograph_cv(image);
    namedWindow("OpencvHistImage");
    imshow("OpencvHistImage", outimage_cv);

	outimage_cpp = getHistograph_cpp(image);
	namedWindow("IntensityTransHistImage", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTransHistImage", outimage_cpp);
}

void IntensityTrans::getLinetrans()
{
	// Mat outimage(image.rows, image.cols, CV_8UC1);
	Mat outimage = image.clone();
	double dFa = 2.0, dFb = 0.0;
	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;

	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			uchar gray = image.at<uchar>(i, j);
			int targetgray = dFa * gray + dFb;
			if(targetgray < 0)
				targetgray = 0;
			if(targetgray > 255)
				targetgray = 255;
			outimage.at<uchar>(i, j) = targetgray;
		}
	}

	namedWindow("Linetrans image", CV_WINDOW_AUTOSIZE);
	imshow("Linetrans image", outimage);
}

void IntensityTrans::getLogTrans()
{
	Mat outimage = image.clone();
	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;

	for(int i = 0; i < nRows; i++)
	{
		uchar* pRow = image.ptr<uchar>(i);
		for(int j = 0; j < nCols; j++)
		{
			int grayvalue = image.at<uchar>(i, j);
			
			int targetvalue = log(grayvalue + 1);
			if(targetvalue < 0)
				targetvalue = 0;
			if(targetvalue > 255)
				targetvalue = 255;
			outimage.at<uchar>(i, j) = targetvalue;
		}
	}

	namedWindow("LogTrans image", CV_WINDOW_AUTOSIZE);
	imshow("LogTrans image", outimage);
}

void IntensityTrans::getGamaTrans()
{
	Mat outimage = image.clone();
	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;
	double gama = 1.5;
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			int grayvalue = image.at<uchar>(i, j);
			
			int targetvalue = pow(grayvalue / 255.0, gama) * 255;
			if(targetvalue < 0)
				targetvalue = 0;
			if(targetvalue > 255)
				targetvalue = 255;
			outimage.at<uchar>(i, j) = targetvalue;
		}
	}

	namedWindow("GamaTrans image", CV_WINDOW_AUTOSIZE);
	imshow("GamaTrans image", outimage);
}

void IntensityTrans::getThresholdTrans()
{
	Mat outimage = image.clone();
	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;
	uchar threshold = 100;
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			int grayvalue = image.at<uchar>(i, j);
			int targetvalue = 0;
			if(grayvalue < threshold)
				targetvalue = 0;
			if(grayvalue > threshold)
				targetvalue = 255;
			outimage.at<uchar>(i, j) = targetvalue;
		}
	}

	namedWindow("ThresholdTrans image", CV_WINDOW_AUTOSIZE);
	imshow("ThresholdTrans image", outimage);
}

void IntensityTrans::getSegLineTrans()
{
	Mat outimage = image.clone();
	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;
	uchar x1 = 60, x2 = 150, y1 = 80, y2 = 150;
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			int grayvalue = image.at<uchar>(i, j);
			int targetvalue = 0;
			if(grayvalue < x1)
			{
				targetvalue = grayvalue * (y1 / x1);
			}
			else if(grayvalue <= x2)
			{
				targetvalue = ((y2-y1) / (x2 - x1)) * (grayvalue - x1) + y1;
			}
			else
			{
				targetvalue = ((255 - y2) / (255 - x2)) * (grayvalue - x2) + y2;
			}
			if(targetvalue < 0)
				targetvalue = 0;
			if(targetvalue > 255)
				targetvalue = 255;
			outimage.at<uchar>(i, j) = targetvalue;
		}
	}

	namedWindow("SegLineTrans image", CV_WINDOW_AUTOSIZE);
	imshow("SegLineTrans image", outimage);
}