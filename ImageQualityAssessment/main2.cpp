#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

bool blurDetect(Mat srcImage);

int main2()
{
	//读入图片
	Mat img1 = imread("F:\\立方体4.jpg");

	double time = (double)getTickCount();
	bool flag = blurDetect(img1);
	time = ((double)getTickCount() - time) / getTickFrequency();
	cout << "所用时间为：" << time << "s" << endl;
	system("pause");
	return 0;
}

//模糊检测，如果原图像是模糊图像，返回0，否则返回1
bool blurDetect(Mat srcImage)
{

	Mat gray1;
	if (srcImage.channels() != 1)
	{
		//进行灰度化
		cvtColor(srcImage, gray1, CV_RGB2GRAY);
	}
	else
	{
		gray1 = srcImage.clone();
	}
	Mat tmp_m1, tmp_sd1;	//用来存储均值和方差
	double m1 = 0, sd1 = 0;
	//使用3x3的Laplacian算子卷积滤波
	Laplacian(gray1, gray1, CV_16S, 3);
	//归到0~255
	convertScaleAbs(gray1, gray1);
	//计算均值和方差
	meanStdDev(gray1, tmp_m1, tmp_sd1);
	m1 = tmp_m1.at<double>(0, 0);		//均值
	sd1 = tmp_sd1.at<double>(0, 0);		//标准差
	//cout << "原图像：" << endl;
	cout << "均值: " << m1 << " , 方差: " << sd1 * sd1 << endl;
	if (sd1*sd1 < 400)
	{
		cout << "原图像是模糊图像" << endl;
		return 0;
	}
	else
	{
		cout << "原图像是清晰图像" << endl;
		return 1;
	}
}

