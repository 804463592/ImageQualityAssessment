#pragma once

#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;

Mat combineImages(vector<Mat>imgs,//@parameter1:需要显示的图像组 
	int col,//parameter2:显示的列数
	int row, //parameter3:显示的行数
	bool hasMargin) {//parameter4:是否设置边框
	int imgAmount = imgs.size();//获取需要显示的图像数量

	int width(0), height(0);
	for (int i = 0; i < imgs.size(); i++)
	{
		if (width < imgs[i].cols) width = imgs[i].cols;
		if (height < imgs[i].rows)height = imgs[i].rows;
	}
	//int width = imgs[0].cols;//本函数默认需要显示的图像大小相同
	//int height = imgs[0].rows;//获取图像宽高
	int newWidth, newHeight;//新图像宽高
	if (!hasMargin) {
		newWidth = col * width;//无边框，新图像宽/高=原图像宽/高*列/行数
		newHeight = row * height;
	}
	else {
		newWidth = (col + 1) * 20 + col * width;//有边框，要将上边框的尺寸，这里设置边框为20px
		newHeight = (row + 1) * 20 + row * height;
	}

	Mat newImage(newHeight, newWidth, CV_8UC3, Scalar(255, 255, 255));//显示创建设定尺寸的新的大图像；色深八位三通道；填充为白色

	int x, y, imgCount;//x列号，y行号，imgCount图片序号
	if (hasMargin) {//有边框
		imgCount = 0;
		x = 0; y = 0;
		while (imgCount < imgAmount) {
			Mat imageROI = newImage(Rect(x*width + (x + 1) * 20, y*height + (y + 1) * 20, imgs[imgCount].cols, imgs[imgCount].rows));//创建感兴趣区域
			imgs[imgCount].copyTo(imageROI);//将图像复制到大图中
			imgCount++;
			if (x == (col - 1)) {
				x = 0;
				y++;
			}
			else {
				x++;
			}//移动行列号到下一个位置
		}
	}
	else {//无边框
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
	return newImage;//返回新的组合图像
};
