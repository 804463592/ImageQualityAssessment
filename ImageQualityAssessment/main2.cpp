#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

bool blurDetect(Mat srcImage);

int main2()
{
	//����ͼƬ
	Mat img1 = imread("F:\\������4.jpg");

	double time = (double)getTickCount();
	bool flag = blurDetect(img1);
	time = ((double)getTickCount() - time) / getTickFrequency();
	cout << "����ʱ��Ϊ��" << time << "s" << endl;
	system("pause");
	return 0;
}

//ģ����⣬���ԭͼ����ģ��ͼ�񣬷���0�����򷵻�1
bool blurDetect(Mat srcImage)
{

	Mat gray1;
	if (srcImage.channels() != 1)
	{
		//���лҶȻ�
		cvtColor(srcImage, gray1, CV_RGB2GRAY);
	}
	else
	{
		gray1 = srcImage.clone();
	}
	Mat tmp_m1, tmp_sd1;	//�����洢��ֵ�ͷ���
	double m1 = 0, sd1 = 0;
	//ʹ��3x3��Laplacian���Ӿ���˲�
	Laplacian(gray1, gray1, CV_16S, 3);
	//�鵽0~255
	convertScaleAbs(gray1, gray1);
	//�����ֵ�ͷ���
	meanStdDev(gray1, tmp_m1, tmp_sd1);
	m1 = tmp_m1.at<double>(0, 0);		//��ֵ
	sd1 = tmp_sd1.at<double>(0, 0);		//��׼��
	//cout << "ԭͼ��" << endl;
	cout << "��ֵ: " << m1 << " , ����: " << sd1 * sd1 << endl;
	if (sd1*sd1 < 400)
	{
		cout << "ԭͼ����ģ��ͼ��" << endl;
		return 0;
	}
	else
	{
		cout << "ԭͼ��������ͼ��" << endl;
		return 1;
	}
}

