#pragma once

#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;

Mat combineImages(vector<Mat>imgs,//@parameter1:��Ҫ��ʾ��ͼ���� 
	int col,//parameter2:��ʾ������
	int row, //parameter3:��ʾ������
	bool hasMargin) {//parameter4:�Ƿ����ñ߿�
	int imgAmount = imgs.size();//��ȡ��Ҫ��ʾ��ͼ������

	int width(0), height(0);
	for (int i = 0; i < imgs.size(); i++)
	{
		if (width < imgs[i].cols) width = imgs[i].cols;
		if (height < imgs[i].rows)height = imgs[i].rows;
	}
	//int width = imgs[0].cols;//������Ĭ����Ҫ��ʾ��ͼ���С��ͬ
	//int height = imgs[0].rows;//��ȡͼ����
	int newWidth, newHeight;//��ͼ����
	if (!hasMargin) {
		newWidth = col * width;//�ޱ߿���ͼ���/��=ԭͼ���/��*��/����
		newHeight = row * height;
	}
	else {
		newWidth = (col + 1) * 20 + col * width;//�б߿�Ҫ���ϱ߿�ĳߴ磬�������ñ߿�Ϊ20px
		newHeight = (row + 1) * 20 + row * height;
	}

	Mat newImage(newHeight, newWidth, CV_8UC3, Scalar(255, 255, 255));//��ʾ�����趨�ߴ���µĴ�ͼ��ɫ���λ��ͨ�������Ϊ��ɫ

	int x, y, imgCount;//x�кţ�y�кţ�imgCountͼƬ���
	if (hasMargin) {//�б߿�
		imgCount = 0;
		x = 0; y = 0;
		while (imgCount < imgAmount) {
			Mat imageROI = newImage(Rect(x*width + (x + 1) * 20, y*height + (y + 1) * 20, imgs[imgCount].cols, imgs[imgCount].rows));//��������Ȥ����
			imgs[imgCount].copyTo(imageROI);//��ͼ���Ƶ���ͼ��
			imgCount++;
			if (x == (col - 1)) {
				x = 0;
				y++;
			}
			else {
				x++;
			}//�ƶ����кŵ���һ��λ��
		}
	}
	else {//�ޱ߿�
		imgCount = 0;
		x = 0; y = 0;
		while (imgCount < imgAmount) {
			Mat imageROI = newImage(Rect(x*width, y*height, imgs[imgCount].cols, imgs[imgCount].rows));
			imgs[imgCount].copyTo(imageROI);
			imgCount++;
			if (x == (col - 1)) {
				x = 0;
				y++;
			}
			else {
				x++;
			}
		}
	}
	return newImage;//�����µ����ͼ��
};
