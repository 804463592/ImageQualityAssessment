
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>

#include<iostream>
using namespace std;
using namespace cv;

#include<vector>
#include<string>
#include <io.h>
#include <fstream>
//#include"ConbineImg.h"
#include<math.h>

Mat Rgb2YIQ(Mat&frame)
{
	//输入rgb的mat型图
	Mat dst = frame.clone();
	frame.convertTo(frame, CV_32FC3);
	dst.convertTo(dst, CV_32FC3);
	//逐行逐列逐像素点的转换，虽然at速度不快，但这里计算量不算大
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++) {
			//注意顺序是BGR，因此YIQ对应RGB，QIY 对应于BGR
			dst.at<Vec3f>(i, j)[2] = saturate_cast<float>(((0.299*frame.at<Vec3f>(i, j)[2] +
				0.587*frame.at<Vec3f>(i, j)[1] +
				0.114*frame.at<Vec3f>(i, j)[0])) / 255);
			dst.at<Vec3f>(i, j)[1] = saturate_cast<float>(((0.596*frame.at<Vec3f>(i, j)[2] +
				-0.274*frame.at<Vec3f>(i, j)[1] +
				-0.322*frame.at<Vec3f>(i, j)[0])) / 255);
			dst.at<Vec3f>(i, j)[0] = saturate_cast<float>(((0.211*frame.at<Vec3f>(i, j)[2] +
				-0.523*frame.at<Vec3f>(i, j)[1] +
				0.312*frame.at<Vec3f>(i, j)[0])) / 255) * 200;
		}
	}
	return dst;//返回YIQ形式的mat
}

double  yiqIQA(Mat& igd, Mat&  igr)
{

	Mat imgd = igd.clone();
	Mat imgr = igr.clone();
	//转换到YIQ空间
	Mat yiqr = Rgb2YIQ(imgr);
	Mat yiqd = Rgb2YIQ(imgd);

	//分离Y分量
	Mat mvr[3];
	split(yiqr, mvr);
	Mat yiqr_Y = mvr[2];   //Y分量顺序对应R的顺序
	Mat mvd[3];
	split(yiqd, mvd);
	Mat yiqd_Y = mvd[2];

	//准备初始计算量
	Mat u_r, u_d;   //参考图像和失真图像均值
	Mat sigma_r2;   //参考图像方差
	Mat sigma_rd;  //协方差

	GaussianBlur(yiqr_Y, u_r, Size(11, 11), 1.5);
	GaussianBlur(yiqd_Y, u_d, Size(11, 11), 1.5);

	Mat yiqr_Y2 = yiqr_Y.mul(yiqr_Y);
	Mat u_r2 = u_r.mul(u_r);
	GaussianBlur(yiqr_Y2, sigma_r2, Size(11, 11), 1.5);  //sigma_r2 =E(x^2)
	sigma_r2 -= u_r2;   //sigma_r2 =E(x^2) - (E(x))^2

	Mat yiqr_Y_yiqd_Y = yiqr_Y.mul(yiqd_Y);
	Mat ur_ud;
	ur_ud = u_r.mul(u_d);
	GaussianBlur(yiqr_Y_yiqd_Y, sigma_rd, Size(11, 11), 1.5);  //sigma_rd =E(xy)
	sigma_rd -= ur_ud;	  //cov(x,y) =E(xy) -E(x)E(y) 

	u_r.convertTo(u_r, CV_32FC1);
	u_d.convertTo(u_d, CV_32FC1);
	sigma_r2.convertTo(sigma_r2, CV_32FC1);
	sigma_rd.convertTo(sigma_rd, CV_32FC1);

	//u_r,u_d,sigma_r2,sigma_rd全部存为数组，用于计算
	int width = u_r.cols;
	int height = u_d.rows;

	double** ur_arr = new double*[height];
	double** ud_arr = new double*[height];
	double** sigmar2_arr = new double*[height];
	double** sigmard_arr = new double*[height];

	for (int i = 0; i < height; i++)
	{
		ur_arr[i] = new double[width];
		ud_arr[i] = new double[width];
		sigmar2_arr[i] = new double[width];
		sigmard_arr[i] = new double[width];
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			ur_arr[i][j] = u_r.at<float>(i, j);  //如果使用uchar，8位，不能存小数 

			ud_arr[i][j] = u_d.at<float>(i, j);
			sigmar2_arr[i][j] = sigma_r2.at<float>(i, j);
			sigmard_arr[i][j] = sigma_rd.at<float>(i, j);
		
		}
	}
	//计算参考图像与失真图像的亮度变化ls_lc
	double **ls_lc = new double*[height];
	for (int i = 0; i < height; i++)
	{
		ls_lc[i] = new double[width];
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{  //alpha1=1,alpha2 =1;
			ls_lc[i][j] = exp(-abs(ur_arr[i][j] - ud_arr[i][j]) / 255)  * log(1 + (sigmard_arr[i][j] + 0.001) / (sigmar2_arr[i][j] + 0.001));
		}
	}
	/*色度失真因子, 当对比度变化较大时也会引起色彩的失真*/
	//分离I，Q分量
	Mat yiqr_I = mvr[1];   //Y分量顺序对应R的顺序,YIQ<->RGB
	Mat yiqd_I = mvd[1];

	Mat yiqr_Q = mvr[0];
	Mat yiqd_Q = mvd[0];

	//转换 格式,不过貌似本来的格式就是CV_32FC1
	yiqr_I.convertTo(yiqr_I, CV_32FC1);
	yiqd_I.convertTo(yiqd_I, CV_32FC1);
	yiqr_Q.convertTo(yiqr_Q, CV_32FC1);
	yiqd_Q.convertTo(yiqd_Q, CV_32FC1);

	double** CIQ = new double*[height];

	for (int i = 0; i < height; i++)
	{
		CIQ[i] = new double[width];
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double Ir = yiqr_I.at<float>(i, j);
			double Id = yiqd_I.at<float>(i, j);
			double Qr = yiqr_Q.at<float>(i, j);
			double Qd = yiqd_Q.at<float>(i, j);
			//色度失真因子Ciq
			CIQ[i][j] = (2 * Ir*Id + 0.01)*(2 * Qr*Qd + 0.001) / ((Ir*Ir + Id * Id + 0.01)*(Qr*Qr + Qd * Qd + 0.001));
		}
	}
	/*基于亮度强度的权重图,  选择参考图像和失真图像中亮度较强者作
为权重系数来衡量各个像素对整张图像质量影响*/
	double **w = new double*[height];
	for (int i = 0; i < height; i++)
	{
		w[i] = new double[width];
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			w[i][j] = min(exp(-abs(ur_arr[i][j] / 255)), exp(-abs(ud_arr[i][j] / 255)));
		}
	}
	/*对比度变化的图像质量分数*/
	double yiqIQAsre = 0;
	double sum_w = 0;
	double sum_slw = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			sum_slw += CIQ[i][j] * ls_lc[i][j] * w[i][j];
			sum_w += w[i][j];
		}
	}

	yiqIQAsre = sum_slw / sum_w;

	//TODO：delete :   ur_arr,ud_arr,sigma_r2,sigma_rd等数组
	for (int i = 0; i < height; i++)
	{
		delete[] ur_arr[i];
		delete[] ud_arr[i];
		delete[] sigmar2_arr[i];
		delete[] sigmard_arr[i];
		delete[] ls_lc[i];
		delete[] CIQ[i];
		delete[] w[i];
	}
	delete[]ur_arr;
	delete[]ud_arr;
	delete[] sigmar2_arr;
	delete[] sigmard_arr;
	delete[] ls_lc;
	delete[] CIQ;
	delete[] w;

	//printf("CCIQA:%.3f", yiqIQAsre);
	return yiqIQAsre;
}

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
	if (hasMargin) {//有边框，为固定值20
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

/*计算灰度图的均值和方差，实际上后面并没有用到该函数*/
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
	//形状不一样直接退出程序
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
		return 100;
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
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);  //sigma1_2  =E(x^2)
	sigma1_2 -= mu1_2;   //sigma1_2:  sigma_x_2 = E(x^2) -(E(x))^2
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);  //cov(x,y) =E(xy)-E(x)E(y)
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

/*
*功能：将文字显示在图片上
*imageSource:源图，meanValue和attention_Str:要显示的分数和分数类型；x,y,white_width,white_height显示的白色区域的起始点坐标，宽度高度
*/
void putTextOnImg(Mat& imageSource, double meanValue, string attention_Str = "yiqIQA: ", int x = 20, int y = 50, int white_width = 230, int white_height = 30)
{
	Mat newImage(white_height,white_width, CV_8UC3, Scalar(255, 255, 255));//height * width,色深八位三通道；填充为白色
	Mat imageROI = imageSource(Rect(x,y-23, white_width,white_height));//x，y，width * height, 向右为x，向下为y创建原图中的感兴趣区域.
	newImage.copyTo(imageROI);

	////也可以直接遍历元素，将putText的区域变为白色
	//for (int i = y-23; i < y+ white_height; i++)  
	//	for (int j =x; j < x+ white_width; j++)
	//	{
	//		imageSource.at<Vec3b>(i, j)[0] = 255; //i 是rows，j是cols,也就是 行和列
	//		imageSource.at<Vec3b>(i, j)[1] = 255;
	//		imageSource.at<Vec3b>(i, j)[2] = 255;
	//	}
//	imshow("imagesoure:", imageSource);
//	waitKey();

	//double to string
	stringstream meanValueStream;
	string meanValueString;
	meanValueStream << meanValue;
	meanValueStream >> meanValueString;
	meanValueString = attention_Str + meanValueString;
	//putText on image
	putText(imageSource, meanValueString, Point(x, y), CV_FONT_HERSHEY_COMPLEX, 0.8, Scalar(25, 25, 255), 2);
}

double imgQualityAssess(Mat& img)
{
	double score =0;
	double meanStdVal = meanStdValCount(img);  //标准差
	double gradValue = gradCount(img, "Sobel");  //平均梯度
	double entropyVal = entropy(img);   //信息熵
	//double meanGrad = meanGradient(img);
	score = 0.4*meanStdVal+ 0.35*gradValue +0.25*entropyVal ;
	return score;
}

void runDefalut()
{//读取当前可执行文件所在的文件夹内的jpg图片
	string *imgFilePath = new string[5];
	for (int i = 0; i < 5; i++)
	{
		string str = "img";
		imgFilePath[i] = str + std::to_string(i + 1) + ".jpg";
	}
	vector<Mat> imgVec;
	for (int k = 0; k < 5; k++)
	{
		Mat src = imread(imgFilePath[k]);
		imgVec.push_back(src);

	}

	//yiqIQA score
	vector<double> scoreVec;
	for (int i = 0; i < imgVec.size(); i++)
	{
		//double score = imgQualityAssess(imgVec[i]);
		double score = yiqIQA(imgVec[i], imgVec[imgVec.size() - 1]);
		scoreVec.push_back(score);
		putTextOnImg(imgVec[i], score);
	}
	//PSNR:PSNR值越大，表明待评图像与参考图像之间的失真较小，图像质量较好。
	vector<double> psnrScoreVec;
	Mat  referenceImg = imgVec[imgVec.size() - 1];
	for (int i = 0; i < imgVec.size(); i++)
	{
		double mse = getMSE(imgVec[i], referenceImg);
		double score = getPSNR(imgVec[i], referenceImg, mse);
		psnrScoreVec.push_back(score);
		putTextOnImg(imgVec[i], score, "PSNR :", 20, 80);
	}

	//SSIM:SSIM值越大，图像质量越好
	vector<double> ssimScoreVec;
	Mat referImg = imgVec[imgVec.size() - 1];
	for (int i = 0; i < imgVec.size(); i++)
	{
		double ssimScore = getMSSIM(imgVec[i], referImg);
		ssimScoreVec.push_back(ssimScore);
		putTextOnImg(imgVec[i], ssimScore, "SSIM :", 20, 110);
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
	//TODO:暂时不考虑搜索特定的文件类型，我们先默认该文件夹内只有图片类型，没有其他比如txt
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

	int x = 10, y = 10;  //结果显示区域左上角的起点
	for (int i = 0; i < imgVec.size(); i++)
	{
		if(!isSizeSame) {

			double meanStdVal = meanStdValCount(imgVec[i]);  //标准差
			double gradValue = gradCount(imgVec[i], "Sobel");  //平均梯度
			double entropyVal = entropy(imgVec[i]);   //信息熵
			//double myIQAsre;
			//myIQAsre = imgQualityAssess(imgVec[i]);
			putTextOnImg(imgVec[i], meanStdVal, "meanStd:", x, y+30,250);
			putTextOnImg(imgVec[i], gradValue, "meanGrad:", x, y + 60,250);
			putTextOnImg(imgVec[i], entropyVal, "entropyVal:", x, y + 90,250);
		}
		else
		{
			double yiqiqa = yiqIQA(imgVec[i], imgVec[imgVec.size() - 1]);
			putTextOnImg(imgVec[i], yiqiqa, "yiqIQA:", x, y+30);

			double psnrSre, ssimSre;
			double mse = getMSE(imgVec[i], imgVec[imgVec.size() - 1]);
			psnrSre = getPSNR(imgVec[i], imgVec[imgVec.size() - 1], mse);
			putTextOnImg(imgVec[i], psnrSre, "PSNR:", x, y+60);

			ssimSre = getMSSIM(imgVec[i], imgVec[imgVec.size() - 1]);
			putTextOnImg(imgVec[i], ssimSre, "SSIM:", x, y+90);
		}
		//text the img name
	 //  for (int p = 0; p < 40; p++)  
		//     for (int q =0; q <500; q++)
		//    {
		//	imgVec[i].at<Vec3b>(p, q)[0] = 255; //p 是rows，q是cols,也就是 行和列
		//	imgVec[i].at<Vec3b>(p, q)[1] = 255;
		//	imgVec[i].at<Vec3b>(p, q)[2] = 255;
		//     }
		//putText(imgVec[i], imgPathVec[i], Point(0, 30), FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 25, 25), 2);
	}
	//TODO：考虑根据图片数量，生成如何排列图片的列数和行数，cols,rows,
	int n= imgVec.size();
	int cols = 4, rows = 3;
	if (0 > n && n <= 4)
	{
		cols = 2; rows = 2;
	}
	else if (n >= 5 && n <= 8)
	{
		if (n <= 6)
		{
			cols = 3, rows = 2;
		}
		else {
			cols = 4; rows = 2;
		}
	
	}
	else if (n >= 9 && n <= 12)
	{
		if (n == 9) { cols = 3; rows = 3; }
		cols = 4; rows = 3;
	}
	else {
		//s.t.  cols * rows >n
		cols = int(sqrt(n)) + 1;
		rows = cols;
	}

	Mat img = combineImages(imgVec, cols, rows, true);
	//imwrite("Img.jpg", img);   //保存图片
	namedWindow("testImg", 0);
	imshow("testImg", img);
	waitKey();
}

int main()
{
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
				cout << "Default program is running..." << endl;
				runDefalut();
			}
			else if (str == "runUserPath" or str =="ru")
			{//运行用户自定义的数据，目前暂时认为所有图片大小一致,如果不一致，则全部按照最大的图片尺寸拼接在一起
				cout << "Please input img path（eg：F://testImg//aligned_Img）:" << endl;
				string imgPath;
				//imgPath = "F://testImg//aligned_Img";
				cin >> imgPath;
				cout << "path:" << imgPath << endl;
				cout << "Program is running..." << endl;
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
