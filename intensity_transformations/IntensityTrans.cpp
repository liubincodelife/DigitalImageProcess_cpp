#include "IntensityTrans.h"
#include <math.h>
#include <string>

IntensityTrans::IntensityTrans(Mat &input_img)
{
    image = input_img.clone();

	int nChannels = image.channels();
	int nRows = image.rows;
	int nCols = image.cols * nChannels;
	
	// cout<<"nChannels = "<<nChannels<<endl;
	// cout<<"nRows = "<<nRows<<endl;
	// cout<<"nCols = "<<nCols<<endl;

	// namedWindow("IntensityTransSrcImage", CV_WINDOW_AUTOSIZE);
    // imshow("IntensityTransSrcImage", image);
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

	cout<<"cpp nchannels = "<<nChannels<<endl;
	cout<<"cpp nRows = "<<nRows<<endl;
	cout<<"cpp nCols = "<<nCols<<endl;
	
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
	Mat graymat(256, 1, CV_32SC1, graycalc);
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
	Mat srcimage1 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(4)(bottom_left).tif", 0);
	Mat srcimage2 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(1)(top_left).tif", 0);
	Mat srcimage3 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(2)(2nd_from_top).tif", 0);
	Mat srcimage4 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(3)(third_from_top).tif", 0);
	Mat outimage_cv;
	Mat outimage_cpp;
	if(!srcimage1.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	if(!srcimage2.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	if(!srcimage3.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	if(!srcimage4.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	int nChannels = srcimage1.channels();
	int nRows = srcimage1.rows;
	int nCols = srcimage1.cols * nChannels;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;
	
	//srcimage1 hist trans
    namedWindow("IntensityTrans SrcImage1", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage1", srcimage1);
	
    outimage_cv = getHistograph_cv(srcimage1);
    namedWindow("Opencv HistImage1");
    imshow("Opencv HistImage1", outimage_cv);

	outimage_cpp = getHistograph_cpp(srcimage1);
	namedWindow("IntensityTrans HistImage1", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistImage1", outimage_cpp);
	
	//srcimage2 hist trans
	namedWindow("IntensityTrans SrcImage2", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage2", srcimage2);
	
    outimage_cv = getHistograph_cv(srcimage2);
    namedWindow("Opencv HistImage2");
    imshow("Opencv HistImage2", outimage_cv);

	outimage_cpp = getHistograph_cpp(srcimage2);
	namedWindow("IntensityTrans HistImage2", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistImage2", outimage_cpp);

	//srcimage3 hist trans
	namedWindow("IntensityTrans SrcImage3", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage3", srcimage3);
	
    outimage_cv = getHistograph_cv(srcimage3);
    namedWindow("Opencv HistImage3");
    imshow("Opencv HistImage3", outimage_cv);

	outimage_cpp = getHistograph_cpp(srcimage3);
	namedWindow("IntensityTrans HistImage3", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistImage3", outimage_cpp);

	//srcimage4 hist trans
	namedWindow("IntensityTrans SrcImage4", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage4", srcimage4);
	
    outimage_cv = getHistograph_cv(srcimage4);
    namedWindow("Opencv HistImage4");
    imshow("Opencv HistImage4", outimage_cv);

	outimage_cpp = getHistograph_cpp(srcimage4);
	namedWindow("IntensityTrans HistImage4", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistImage4", outimage_cpp);	
}

void getHistEqualize_cv(Mat &grayimage, Mat &dsthistimage)
{
	if(!grayimage.data)
	{
		cout<<"getHistEqualize_cv::input image is empty!!!"<<endl;
		return;
	}
	equalizeHist(grayimage, dsthistimage);
}

void getHistEqualize_cpp(Mat& grayimage, Mat& outimage)
{
	if(!grayimage.data)
	{
		cout<<"getHistEqualize_cpp::grayimage is empty!!!"<<endl;
		return;
	}
	int nChannels = grayimage.channels();
	int nRows = grayimage.rows;
	int nCols = grayimage.cols * nChannels;

	int gray[256] = {0};
	double gray_prob[256] = {0};
	double gray_distribution[256] = {0};
	int gray_equlize[256] = {0};
	int pixel_sum = nRows * nCols;

	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			uchar grayvalue = grayimage.at<uchar>(i, j);
			gray[grayvalue]++;
		}
	}

	for(int i = 0; i < 256; i++)
	{
		gray_prob[i] = (double)gray[i] / pixel_sum;
	}

	gray_distribution[0] = gray_prob[0];
	for(int i = 1; i < 256; i++)
	{
		gray_distribution[i] = gray_distribution[i-1] + gray_prob[i];
	}

	for(int i = 0; i < 256; i++)
	{
		gray_equlize[i] = uchar(255 * gray_distribution[i] + 0.5);
	}

	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			int tempvalue = grayimage.at<uchar>(i, j);
			outimage.at<uchar>(i, j) = gray_equlize[tempvalue];
		}
	}
}

void IntensityTrans::getHistEqualize()
{
	Mat srcimage1 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(4)(bottom_left).tif", 0);
	Mat srcimage2 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(1)(top_left).tif", 0);
	Mat srcimage3 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(2)(2nd_from_top).tif", 0);
	Mat srcimage4 = imread("../images/DIP3E_Original_Images_CH03/Fig0316(3)(third_from_top).tif", 0);
	Mat outimage_cv = srcimage1.clone();
	Mat outimage_cpp = srcimage1.clone();;
	if(!srcimage1.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	if(!srcimage2.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	if(!srcimage3.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	if(!srcimage4.data)
	{
		cout<<"srcimage1 is empty!!!"<<endl;
		return;
	}

	int nChannels = srcimage1.channels();
	int nRows = srcimage1.rows;
	int nCols = srcimage1.cols * nChannels;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;
	
	//srcimage1 histequlize trans
    namedWindow("IntensityTrans SrcImage1", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage1", srcimage1);
	
    getHistEqualize_cv(srcimage1, outimage_cv);
    namedWindow("Opencv HistEqulizeImage1");
    imshow("Opencv HistEqulizeImage1", outimage_cv);

	getHistEqualize_cpp(srcimage1, outimage_cpp);
	namedWindow("IntensityTrans HistEqualizeImage1", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistEqualizeImage1", outimage_cpp);

	Mat outhistimage_cv = getHistograph_cv(outimage_cv);
    namedWindow("Opencv HistImage1");
    imshow("Opencv HistImage1", outhistimage_cv);

	Mat outhistimage_cpp = getHistograph_cpp(outimage_cpp);
	namedWindow("IntensityTrans HistImage1", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistImage1", outhistimage_cpp);
	
	//srcimage2 histequlize trans
	namedWindow("IntensityTrans SrcImage2", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage2", srcimage2);
	
    getHistEqualize_cv(srcimage2, outimage_cv);
    namedWindow("Opencv HistEqulizeImage2");
    imshow("Opencv HistEqulizeImage2", outimage_cv);

	getHistEqualize_cpp(srcimage2, outimage_cpp);
	namedWindow("IntensityTrans HistEqulizeImage2", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistEqulizeImage2", outimage_cpp);

	//srcimage3 histequlize trans
	namedWindow("IntensityTrans SrcImage3", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage3", srcimage3);
	
    getHistEqualize_cv(srcimage3, outimage_cv);
    namedWindow("Opencv HistEqulizeImage3");
    imshow("Opencv HistEqulizeImage3", outimage_cv);

	getHistEqualize_cpp(srcimage3, outimage_cpp);
	namedWindow("IntensityTrans HistEqulizeImage3", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistEqulizeImage3", outimage_cpp);

	//srcimage4 histequlize trans
	namedWindow("IntensityTrans SrcImage4", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans SrcImage4", srcimage4);
	
    getHistEqualize_cv(srcimage4, outimage_cv);
    namedWindow("Opencv HistEqulizeImage4");
    imshow("Opencv HistEqulizeImage4", outimage_cv);

	getHistEqualize_cpp(srcimage4, outimage_cpp);
	namedWindow("IntensityTrans HistEqulizeImage4", CV_WINDOW_AUTOSIZE);
    imshow("IntensityTrans HistEqulizeImage4", outimage_cpp);
}

void IntensityTrans::getReverseTrans()
{
	Mat srcimage = imread("../images/DIP3E_Original_Images_CH03/Fig0304(a)(breast_digital_Xray).tif", 0);
	Mat outimage = srcimage.clone();

	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;

	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			uchar gray = srcimage.at<uchar>(i, j);
			int targetgray = 255 - gray;
			if(targetgray < 0)
				targetgray = 0;
			if(targetgray > 255)
				targetgray = 255;
			outimage.at<uchar>(i, j) = targetgray;
		}
	}
	namedWindow("ReverseTrans srcimage", CV_WINDOW_AUTOSIZE);
	imshow("ReverseTrans srcimage", srcimage);
	namedWindow("ReverseTrans outimage", CV_WINDOW_AUTOSIZE);
	imshow("ReverseTrans outimage", outimage);
}

void IntensityTrans::getLineTrans()
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
	Mat srcimage = imread("../images/DIP3E_Original_Images_CH03/Fig0305(a)(DFT_no_log).tif", 0);
	Mat outimage = srcimage.clone();

	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;

	for(int i = 0; i < nRows; i++)
	{
		uchar* pRow = image.ptr<uchar>(i);
		for(int j = 0; j < nCols; j++)
		{
			float grayvalue = srcimage.at<uchar>(i, j);
			int targetvalue = log(grayvalue + 1) / log(255 + 1) * 255.0; //normalize targetvalue
			if(targetvalue < 0)
				targetvalue = 0;
			if(targetvalue > 255)
				targetvalue = 255;
			outimage.at<uchar>(i, j) = targetvalue;
		}
	}
	// normalize(outimage, outimage, 0, 255, CV_MINMAX);
	// convertScaleAbs(outimage, outimage);

	namedWindow("LogTrans srcimage", CV_WINDOW_AUTOSIZE);
	imshow("LogTrans srcimage", srcimage);
	namedWindow("LogTrans outimage", CV_WINDOW_AUTOSIZE);
	imshow("LogTrans outimage", outimage);
}

void IntensityTrans::getGamaTrans()
{
	//Demo image1
	// Mat srcimage = imread("../images/DIP3E_Original_Images_CH03/Fig0308(a)(fractured_spine).tif");
	//Demo image2
	Mat srcimage = imread("../images/DIP3E_Original_Images_CH03/Fig0309(a)(washed_out_aerial_image).tif", 0);
	Mat outimage1 = srcimage.clone();
	Mat outimage2 = srcimage.clone();
	Mat outimage3 = srcimage.clone();

	int nChannels = srcimage.channels();
	int nRows = srcimage.rows;
	int nCols = srcimage.cols * nChannels;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;
	//parameters for image1
	// double gama1 = 0.6, gama2 = 0.4, gama3 = 0.3;
	//parameters for image2
	double gama1 = 3.0, gama2 = 4.0, gama3 = 5.0;
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			int grayvalue = srcimage.at<uchar>(i, j);
			
			int targetvalue1 = pow(grayvalue / 255.0, gama1) * 255;  //normalize first
			int targetvalue2 = pow(grayvalue / 255.0, gama2) * 255;  //normalize first
			int targetvalue3 = pow(grayvalue / 255.0, gama3) * 255;  //normalize first

			if(targetvalue1 < 0)
				targetvalue1 = 0;
			if(targetvalue1 > 255)
				targetvalue1 = 255;
			if(targetvalue2 < 0)
				targetvalue2 = 0;
			if(targetvalue2 > 255)
				targetvalue2 = 255;
			if(targetvalue3 < 0)
				targetvalue3 = 0;
			if(targetvalue3 > 255)
				targetvalue3 = 255;
				
			outimage1.at<uchar>(i, j) = targetvalue1;
			outimage2.at<uchar>(i, j) = targetvalue2;
			outimage3.at<uchar>(i, j) = targetvalue3;
		}
	}

	namedWindow("GamaTrans srcimage", CV_WINDOW_AUTOSIZE);
	imshow("GamaTrans srcimage", srcimage);
	namedWindow("GamaTrans image(gama=0.6)", CV_WINDOW_AUTOSIZE);
	imshow("GamaTrans image(gama=0.6)", outimage1);
	namedWindow("GamaTrans image(gama=0.4)", CV_WINDOW_AUTOSIZE);
	imshow("GamaTrans image(gama=0.4)", outimage1);
	namedWindow("GamaTrans image(gama=0.3)", CV_WINDOW_AUTOSIZE);
	imshow("GamaTrans image(gama=0.3)", outimage1);
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

//another SegLineTrans function
void dividedLinearStrength(cv::Mat& matInput, cv::Mat& matOutput, float fStart, float fEnd, 
	float fSout, float fEout)
{
	//计算直线参数
	//L1
	float fK1 = fSout / fStart;
	//L2
	float fK2 = (fEout - fSout) / (fEnd - fStart);
	float fC2 = fSout - fK2 * fStart;
	//L3
	float fK3 = (255.0f - fEout) / (255.0f - fEnd);
	float fC3 = 255.0f - fK3 * 255.0f;
 
	//建立查询表
	std::vector<unsigned char> loolUpTable(256);
	for (size_t m = 0; m < 256; m++)
	{
		if (m < fStart)
		{
			loolUpTable[m] = static_cast<unsigned char>(m * fK1);
		}
		else if (m > fEnd)
		{
			loolUpTable[m] = static_cast<unsigned char>(m * fK3 + fC3);
		}
		else
		{
			loolUpTable[m] = static_cast<unsigned char>(m * fK2 + fC2);
		}
	}
	//构造输出图像
	matOutput = cv::Mat::zeros(matInput.rows, matInput.cols, matInput.type());
	//灰度映射
	for (size_t r = 0; r < matInput.rows; r++)
	{
		unsigned char* pInput = matInput.data + r * matInput.step[0];
		unsigned char* pOutput = matOutput.data + r * matOutput.step[0];
		for (size_t c = 0; c < matInput.cols * 3; c++)
		{
			//查表gamma变换
			pOutput[c] = loolUpTable[pInput[c]];
		}
	}
}

void IntensityTrans::getSegLineTrans()
{
	Mat srcimage = imread("../images/DIP3E_Original_Images_CH03/Fig0310(b)(washed_out_pollen_image).tif", 0);
	Mat outimage = srcimage.clone();

	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;

	double minvalue = 0, maxvalue = 0;
	minMaxLoc(srcimage, &minvalue, &maxvalue);

	cout<<"minvalue = "<<minvalue<<endl;
	cout<<"maxvalue = "<<maxvalue<<endl;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;

	// uchar x1 = 104, x2 = 104, y1 = 0, y2 = 255;  //binary transform
	uchar x1 = 104, x2 = 114, y1 = 0, y2 = 200;  //binary transform
	// dividedLinearStrength(srcimage, outimage, x1, x2, y1, y2);
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			int grayvalue = srcimage.at<uchar>(i, j);
			int targetvalue = 0;
			if(x1 == x2)
			{
				if(grayvalue < x1)
					targetvalue = 0;
				if(grayvalue > x2)
					targetvalue = 255;
			}
			else
			{
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
			}
			
			if(targetvalue < 0)
				targetvalue = 0;
			if(targetvalue > 255)
				targetvalue = 255;
			outimage.at<uchar>(i, j) = targetvalue;
		}
	}

	namedWindow("SegLineTrans srcimage", CV_WINDOW_AUTOSIZE);
	imshow("SegLineTrans srcimage", srcimage);
	namedWindow("SegLineTrans outimage", CV_WINDOW_AUTOSIZE);
	imshow("SegLineTrans outimage", outimage);
}

void IntensityTrans::getIntensitySlicingTrans()
{
	Mat srcimage = imread("../images/DIP3E_Original_Images_CH03/Fig0312(a)(kidney).tif", 0);
	Mat outimage = srcimage.clone();

	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;

	double minvalue = 0, maxvalue = 0;
	minMaxLoc(srcimage, &minvalue, &maxvalue);

	cout<<"minvalue = "<<minvalue<<endl;
	cout<<"maxvalue = "<<maxvalue<<endl;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;

	uchar x1 = 150, x2 = 230;
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			float grayvalue = srcimage.at<uchar>(i, j);
			int targetvalue = 0;
			if(grayvalue >= x1 && grayvalue <= x2)
			{
				targetvalue = 255;
			}
			else
			{
				targetvalue = 0;
			}
			if(targetvalue < 0)
				targetvalue = 0;
			if(targetvalue > 255)
				targetvalue = 255;
			outimage.at<uchar>(i, j) = targetvalue;
		}
	}

	namedWindow("getIntensitySlicingTrans srcimage", CV_WINDOW_AUTOSIZE);
	imshow("getIntensitySlicingTrans srcimage", srcimage);
	namedWindow("getIntensitySlicingTrans outimage", CV_WINDOW_AUTOSIZE);
	imshow("getIntensitySlicingTrans outimage", outimage);
}

void IntensityTrans::getBitSlicingTrans()
{
	Mat srcimage = imread("../images/DIP3E_Original_Images_CH03/Fig0314(a)(100-dollars).tif", 0);
	Mat outimage = srcimage.clone();

	int nChannels = outimage.channels();
	int nRows = outimage.rows;
	int nCols = outimage.cols * nChannels;

	double minvalue = 0, maxvalue = 0;
	minMaxLoc(srcimage, &minvalue, &maxvalue);

	cout<<"minvalue = "<<minvalue<<endl;
	cout<<"maxvalue = "<<maxvalue<<endl;

	cout<<"nChannels = "<<nChannels<<endl;
	cout<<"nRows = "<<nRows<<endl;
	cout<<"nCols = "<<nCols<<endl;
	uchar bitlevel = 8;
	for(int k = 0; k < bitlevel; k++)
	{
		for(int i = 0; i < nRows; i++)
		{
			for(int j = 0; j < nCols; j++)
			{
				int grayvalue = srcimage.at<uchar>(i, j);
				int bitlevelmask = 0x01 << k;
				int targetvalue = grayvalue & bitlevelmask;
				if(targetvalue  == bitlevelmask)
				{
					targetvalue = 255;
				}
				else
				{
					targetvalue = 0;
				}
				if(targetvalue < 0)
					targetvalue = 0;
				if(targetvalue > 255)
					targetvalue = 255;
				outimage.at<uchar>(i, j) = targetvalue;
			}
		}
		string windowname = "BitSlicingTrans image level: ";
		windowname = windowname + to_string(k);
		imshow(windowname, outimage);
	}

	//using higher 4 plane rebuild the image
	Mat reoutimage = srcimage.clone();
	for(int i = 0; i < nRows; i++)
	{
		for(int j = 0; j < nCols; j++)
		{
			int grayvalue = srcimage.at<uchar>(i, j);
				int bitlevelmask = 0xFF;
				int targetvalue = grayvalue & bitlevelmask;
				
				if(targetvalue < 0)
					targetvalue = 0;
				if(targetvalue > 255)
					targetvalue = 255;
				reoutimage.at<uchar>(i, j) = targetvalue;
		}
	}

	namedWindow("getIntensitySlicingTrans srcimage", CV_WINDOW_AUTOSIZE);
	imshow("getIntensitySlicingTrans srcimage", srcimage);
	namedWindow("getIntensitySlicingTrans reoutimage", CV_WINDOW_AUTOSIZE);
	imshow("getIntensitySlicingTrans reoutimage", reoutimage);
}