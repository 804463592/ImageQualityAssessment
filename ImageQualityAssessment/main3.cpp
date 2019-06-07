
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>

#include<iostream>
using namespace std;
using namespace cv;

#include<vector>
#include<string>
#include <io.h>
#include <fstream>
#include"ConbineImg.h"


//Mat combineImages(vector<Mat>imgs,//@parameter1:需要显示的图像组 
//	int col,//parameter2:显示的列数
//	int row, //parameter3:显示的行数
//	bool hasMargin) //parameter4:是否设置边框
//{
//	int imgAmount = imgs.size();//获取需要显示的图像数量
//	int width = imgs[0].cols;//本函数默认需要显示的图像大小相同
//	int height = imgs[0].rows;//获取图像宽高
//	int newWidth, newHeight;//新图像宽高
//	if (!hasMargin) {
//		newWidth = col * imgs[0].cols;//无边框，新图像宽/高=原图像宽/高*列/行数
//		newHeight = row * imgs[0].rows;
//	}
//	else {
//		newWidth = (col + 1) * 20 + col * width;//有边框，要将上边框的尺寸，这里设置边框为20px
//		newHeight = (row + 1) * 20 + row * height;
//	}
//
//	Mat newImage(newHeight, newWidth, CV_8UC3, Scalar(255, 255, 255));//显示创建设定尺寸的新的大图像；色深八位三通道；填充为白色
//
//	int x, y, imgCount;//x列号，y行号，imgCount图片序号
//	if (hasMargin) {//有边框
//		imgCount = 0;
//		x = 0; y = 0;
//		while (imgCount < imgAmount) {
//			Mat imageROI = newImage(Rect(x*width + (x + 1) * 20, y*height + (y + 1) * 20, width, height));//创建感兴趣区域
//			imgs[imgCount].copyTo(imageROI);//将图像复制到大图中
//			imgCount++;
//			if (x == (col - 1)) {
//				x = 0;
//				y++;
//			}
//			else {
//				x++;
//			}//移动行列号到下一个位置
//		}
//	}
//	else {//无边框
//		imgCount = 0;
//		x = 0; y = 0;
//		while (imgCount < imgAmount) {
//			Mat imageROI = newImage(Rect(x*width, y*height, width, height));
//			imgs[imgCount].copyTo(imageROI);
//			imgCount++;
//			if (x == (col - 1)) {
//				x = 0;
//				y++;
//			}
//			else {
//				x++;
//			}
//		}
//	}
//	return newImage;//返回新的组合图像
//};


/*
 单幅图像信息熵计算
 定义中，图像的信息熵通常采用灰度图计算
 */
double entropy(Mat & img)
{
	double temp[256] = { 0.0f };
	// 计算每个像素的累积值
	int row = img.rows;
	int col = img.cols;
	for (int r = 0; r < row; r++)
	{
		for (int c = 0; c < col; c++)
		{
			const uchar * i = img.ptr<uchar>(r, c);
			temp[*i] ++;
		}
	}

	// 计算每个像素的概率
	int size = row * col;
	for (int i = 0; i < 256; i++)
	{
		temp[i] = temp[i] / size;
	}

	double result = 0.0f;
	// 计算图像信息熵
	for (int i = 0; i < 256; i++)
	{
		if (temp[i] != 0.0) {
			result += temp[i] * log2(temp[i]);
		}
	}
	return -result;
}

/*
计算平均梯度
梯度的计算应采用灰度图
*/
double meanGradient(Mat & grayImg) {
	if (grayImg.channels() != 1) {
		printf("avgGradient 参数错误，必须输入单通道图！");
		return 0.0;
	}
	// 原灰度图转换成浮点型数据类型
	Mat src;
	grayImg.convertTo(src, CV_64FC1);

	double temp = 0.0f;
	// 由于求一阶差分的边界问题，这里行列都要-1
	int rows = src.rows - 1;
	int cols = src.cols - 1;

	// 根据公式计算平均梯度
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			// 离散的delta就是相邻的离散点的差值
			double dx = src.at<double>(r, c + 1) - src.at<double>(r, c);
			double dy = src.at<double>(r + 1, c) - src.at<double>(r, c);
			double ds = sqrt((dx*dx + dy * dy) / 2);
			temp += ds;
		}
	}
	double imageAVG = temp / (rows*cols);

	return imageAVG;
}

/*计算灰度图的均值和方差*/
void mean_std(const Mat & grayImg, double & mean, double & std) {
	if (grayImg.channels() != 1) {
		printf("mean_std 参数错误，必须输入单通道图！");
		return;
	}
	Mat mat_mean, mat_stddev;
	meanStdDev(grayImg, mat_mean, mat_stddev);
	mean = mat_mean.at<double>(0, 0);
	std = mat_stddev.at<double>(0, 0);
}

double getMSE(const Mat & src1, const Mat & src2)
{
	//形状不一样直接退出
	if (src1.cols != src2.cols || src1.rows != src2.rows)
	{
		cout << "the sizes of these images are not of the same! " << endl;
		exit(0);
	}

	Mat s1;
	absdiff(src1, src2, s1);    // |src1 - src2|
	s1.convertTo(s1, CV_32F);   // 不能在8位矩阵上做平方运算
	s1 = s1.mul(s1);            // |src1 - src2|^2
	Scalar s = sum(s1);         // 叠加每个通道的元素

	double result = 0.0f;
	int ch = s1.channels();
	for (int i = 0; i < ch; i++) {
		// 叠加所有通道
		result += s.val[i];
	}

	if (result <= 1e-10) // 如果值太小就直接等于0
		return 0;
	else
		return result / (ch*s1.total());
}

double getPSNR(const Mat& src1, const Mat& src2, double MSE) {
	if (MSE <= 1e-10) {
		return 10000;
	}
	return 10.0*log10((255 * 255) / MSE);
}

double getMSSIM(const Mat& src1, const Mat& src2)
{
	// 参数由经验公式取得
	// C1=(K1*L)^2, C2=(K2*L)^2, C3=C2/2, 一般地K1=0.01, K2=0.03, L=255（ 是像素值的动态范围，一般都取为255）
	const double C1 = 6.5025, C2 = 58.5225;
	const int TYPE = CV_32F;

	// 不能在单字节类型上计算，范围溢出，需要转换
	Mat I1, I2;
	src1.convertTo(I1, TYPE);
	src2.convertTo(I2, TYPE);

	Mat I2_2 = I2.mul(I2);  // I2^2
	Mat I1_2 = I1.mul(I1);  // I1^2
	Mat I1_I2 = I1.mul(I2); // I1*I2

	// 高斯函数计算图像的均值、方差以及协方差，而不是采用遍历像素点的方式，以换来更高的效率
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar SSIM = mean(ssim_map);
	// 返回三个通道的SSIM的平均值，[0,1]之间
	return (SSIM.val[2] + SSIM.val[1] + SSIM.val[0]) / 3;

}

//Tenengrad梯度方法、Laplacian梯度方法
 double gradCount(Mat& imageSource,string way)
{

	Mat imageGrey;

	cvtColor(imageSource, imageGrey, CV_RGB2GRAY);
	Mat imageSobel;

	if (way == "Laplacian")
	{
		Laplacian(imageGrey, imageSobel, CV_16U);
	}
	else if(way =="Sobel")
	{
		Sobel(imageGrey, imageSobel, CV_16U, 1, 1);
	}
	else {
		cout << "no such ways!" << endl;
		return 0;
	}
	//图像的平均梯度
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];

	return meanValue;
}

//计算标准差：
double meanStdValCount(Mat& imageSource)
{
	//Mat imageSource = imread("F:\\立方体4.jpg");
	//Mat imageSource = imread("F://lenna.bmp");
	Mat imageGrey;

	cvtColor(imageSource, imageGrey, CV_RGB2GRAY);
	Mat meanValueImage;
	Mat meanStdValueImage;

	//求灰度图像的标准差
	meanStdDev(imageGrey, meanValueImage, meanStdValueImage);
	double meanValue = 0.0;
	meanValue = meanStdValueImage.at<double>(0, 0);

	return meanValue;
}


void putTextOnImg(Mat& imageSource, double meanValue,string attention_Str ="My IQA score : ",int x=20,int y=50)
{
	//double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue;
	meanValueStream >> meanValueString;
	meanValueString = attention_Str + meanValueString;
	putText(imageSource, meanValueString, Point(x, y), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(25, 255, 25), 2);
	
}

double imgQualityAssess(Mat& img)
{
	double score =0;
	double meanValue = meanStdValCount(img);
	double gradValue = gradCount(img, "Sobel");
	double entropyVal = entropy(img);
	score = 0.4*meanValue + 0.35*gradValue +0.25*entropyVal;
	return score;
}

void runDefalut()
{
	string *imgFilePath = new string[5];
	for (int i = 0; i < 5; i++)
	{
		string str = "img";
		imgFilePath[i] = str + std::to_string(i + 1) + ".jpg";
		//cout << imgFilePath[i] << endl;
	}
	vector<Mat> imgVec;
	for (int k = 0; k < 5; k++)
	{
		Mat src = imread(imgFilePath[k]);
		imgVec.push_back(src);
		//imshow("fasd", imgVec[k]);
		//waitKey();
	}

	//my IQA score
	vector<double> scoreVec;
	for (int i = 0; i < imgVec.size(); i++)
	{
		double score = imgQualityAssess(imgVec[i]);
		scoreVec.push_back(score);
		putTextOnImg(imgVec[i], score);
	}
	//PSNR,PSNR值越大，表明待评图像与参考图像之间的失真较小，图像质量较好。
	vector<double> psnrScoreVec;
	Mat  referenceImg = imgVec[imgVec.size() - 1];
	for (int i = 0; i < imgVec.size(); i++)
	{
		double mse = getMSE(imgVec[i], referenceImg);
		double score = getPSNR(imgVec[i], referenceImg, mse);
		psnrScoreVec.push_back(score);
		putTextOnImg(imgVec[i], score, "PSNR score:", 20, 80);
	}

	//SSIM:SSIM值越大，图像质量越好
	vector<double> ssimScoreVec;
	Mat referImg = imgVec[imgVec.size() - 1];
	for (int i = 0; i < imgVec.size(); i++)
	{
		double ssimScore = getMSSIM(imgVec[i], referImg);
		ssimScoreVec.push_back(ssimScore);
		putTextOnImg(imgVec[i], ssimScore, "SSIM score:", 20, 110);
	}
	Mat tImg = combineImages(imgVec, 2, 3, true);
	namedWindow("testImg", 0);
	cvResizeWindow("testImg", 800, 600);
	imshow("testImg", tImg);
	waitKey();
	delete[]imgFilePath;
}
void getAllFiles(string path, vector<string>& files,string fileType =".png")
{
	//文件句柄
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) { //比较文件类型是否是文件夹
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					//递归搜索
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0); //寻找下一个，成功返回0，否则-1
		_findclose(hFile);
	}
}

void runByUserPath(string path)
{
	bool isSizeSame(true);

	//string *imgfilePath = new string[n];
	vector<string> imgPathVec;
	//TODO:后续考虑只获取该路径
	getAllFiles(path, imgPathVec);
	vector<Mat> imgVec;
	for (int i = 0; i < imgPathVec.size(); i++)
	{
		cout << imgPathVec[i] << endl;
		imgVec.push_back(imread(imgPathVec[i]));
	}
	if (imgVec.empty())
	{
		cout << "the directory contains no images!" << endl;
		return;
	}
	//TODO:程序判断图片形状是否一致
	int tem_col(imgVec[0].cols), tem_row(imgVec[0].rows);
	for (int i = 1; i < imgVec.size(); i++)
	{
		if (tem_col != imgVec[i].cols || tem_row != imgVec[i].rows)
		{
			isSizeSame = false;
			break;
		}
	}

	for (int i = 0; i < imgVec.size(); i++)
	{
		double myIQAsre, psnrSre, ssimSre;
		myIQAsre = imgQualityAssess(imgVec[i]);
		putTextOnImg(imgVec[i], myIQAsre, "my IQA:", 20, 50);

		if (isSizeSame==true)
		{
			double mse = getMSE(imgVec[i], imgVec[imgVec.size() - 1]);
			psnrSre = getPSNR(imgVec[i], imgVec[imgVec.size() - 1], mse);
			putTextOnImg(imgVec[i], psnrSre, "PSNR:", 20, 80);

			ssimSre = getMSSIM(imgVec[i], imgVec[imgVec.size() - 1]);
			putTextOnImg(imgVec[i], ssimSre, "SSIM:", 20, 110);
		}
	}
	//TODO：考虑如何根据图片数量，自动生成如何排列图片的列数和行数，cols,rows,
	int cols = 4, rows = 2;
	//int col = int(sqrt(imgVec.size()));

	Mat img = combineImages(imgVec, cols, rows, false);
	namedWindow("testImg", 0);
	imshow("testImg", img);
	waitKey();
}

int main()
{
	//vector<string> temp;
	//getAllFiles("F:\\testImg", temp);
	//for (int i = 0; i < temp.size(); ++i)
	//{
	//	cout << temp[i] << endl;
	//}

	string str;
	cout << "please enter instruction！enter help for help" << endl;
	bool ctrl_flag = true;
	while (cin>>str)
 	{
		    //命令行输入
			if (str == "help" or str =="h")
			{
				cout << "runDefault : Assess the images of the default directory;" << endl;
				cout << "runUserPath：Assess the images of the user's directory" << endl;
				cout << "runSingle:Assess the single image" << endl;
				cout << "cls:clear the console" << endl;
				cout << "quit:exit the console" << endl;
			}
			else if (str == "runDefault" or str =="rd")  
			{//运行默认的数据
				cout << "Default running..." << endl;
				runDefalut();
			}
			else if (str == "runUserPath" or str =="ru")
			{//运行用户自定义的数据，目前暂时认为所有图片大小一致
				cout << "plz insert img path（eg:F://testImg）:" << endl;
				string imgPath;
				imgPath = "F:\\testImg2";
				//cin >> imgPath;
				cout << "path:" << imgPath << endl;
				//string isSizeSame;
				//bool isSame;
				//cout << "Are the sizes of the images the same? true or false" << endl;
				//cin >> isSizeSame;  //bool变量无法通过cin直接输入
				//while (1)
				//{
				//	if (isSizeSame == "true")
				//	{
				//		isSame = true;
				//		break;
				//	}
				//	else if (isSizeSame == "false")
				//	{
				//		isSame = false;
				//		break;
				//	}
				//	else {
				//		cout << "plz reinput" << endl;
				//	}
				//}
	
				runByUserPath(imgPath);
			}
			else if (str == "runSingle" or str =="rs")
			{//测试单张图片
				cout<<"comming soon!"<<endl;
			 //TODO:runSingle

			}
			else if (str == "cls")
			{ //清理控制台
				cout << "clearing the console..." << endl;
				system("cls");
			}
			else if (str == "quit")
			{
				break;
			}
			else cout << "no such instruction!" <<endl;
	}

	system("pause");
	return 0;
}
