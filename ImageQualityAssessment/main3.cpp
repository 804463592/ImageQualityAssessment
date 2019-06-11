
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
	//����rgb��mat��ͼ
	Mat dst = frame.clone();
	frame.convertTo(frame, CV_32FC3);
	dst.convertTo(dst, CV_32FC3);
	//�������������ص��ת������Ȼat�ٶȲ��죬����������������
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++) {
			//ע��˳����BGR�����YIQ��ӦRGB��QIY ��Ӧ��BGR
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
	return dst;//����YIQ��ʽ��mat
}

double  yiqIQA(Mat& igd, Mat&  igr)
{

	Mat imgd = igd.clone();
	Mat imgr = igr.clone();
	//ת����YIQ�ռ�
	Mat yiqr = Rgb2YIQ(imgr);
	Mat yiqd = Rgb2YIQ(imgd);

	//����Y����
	Mat mvr[3];
	split(yiqr, mvr);
	Mat yiqr_Y = mvr[2];   //Y����˳���ӦR��˳��
	Mat mvd[3];
	split(yiqd, mvd);
	Mat yiqd_Y = mvd[2];

	//׼����ʼ������
	Mat u_r, u_d;   //�ο�ͼ���ʧ��ͼ���ֵ
	Mat sigma_r2;   //�ο�ͼ�񷽲�
	Mat sigma_rd;  //Э����

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

	//u_r,u_d,sigma_r2,sigma_rdȫ����Ϊ���飬���ڼ���
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
			ur_arr[i][j] = u_r.at<float>(i, j);  //���ʹ��uchar��8λ�����ܴ�С�� 

			ud_arr[i][j] = u_d.at<float>(i, j);
			sigmar2_arr[i][j] = sigma_r2.at<float>(i, j);
			sigmard_arr[i][j] = sigma_rd.at<float>(i, j);
		
		}
	}
	//����ο�ͼ����ʧ��ͼ������ȱ仯ls_lc
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
	/*ɫ��ʧ������, ���Աȶȱ仯�ϴ�ʱҲ������ɫ�ʵ�ʧ��*/
	//����I��Q����
	Mat yiqr_I = mvr[1];   //Y����˳���ӦR��˳��,YIQ<->RGB
	Mat yiqd_I = mvd[1];

	Mat yiqr_Q = mvr[0];
	Mat yiqd_Q = mvd[0];

	//ת�� ��ʽ,����ò�Ʊ����ĸ�ʽ����CV_32FC1
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
			//ɫ��ʧ������Ciq
			CIQ[i][j] = (2 * Ir*Id + 0.01)*(2 * Qr*Qd + 0.001) / ((Ir*Ir + Id * Id + 0.01)*(Qr*Qr + Qd * Qd + 0.001));
		}
	}
	/*��������ǿ�ȵ�Ȩ��ͼ,  ѡ��ο�ͼ���ʧ��ͼ�������Ƚ�ǿ����
ΪȨ��ϵ���������������ض�����ͼ������Ӱ��*/
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
	/*�Աȶȱ仯��ͼ����������*/
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

	//TODO��delete :   ur_arr,ud_arr,sigma_r2,sigma_rd������
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
	if (hasMargin) {//�б߿�Ϊ�̶�ֵ20
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

/*
 ����ͼ����Ϣ�ؼ���
 �����У�ͼ�����Ϣ��ͨ�����ûҶ�ͼ����
 */
double entropy(Mat & img)
{
	double temp[256] = { 0.0f };
	// ����ÿ�����ص��ۻ�ֵ
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

	// ����ÿ�����صĸ���
	int size = row * col;
	for (int i = 0; i < 256; i++)
	{
		temp[i] = temp[i] / size;
	}

	double result = 0.0f;
	// ����ͼ����Ϣ��
	for (int i = 0; i < 256; i++)
	{
		if (temp[i] != 0.0) {
			result += temp[i] * log2(temp[i]);
		}
	}
	return -result;
}

/*
����ƽ���ݶ�
�ݶȵļ���Ӧ���ûҶ�ͼ
*/
double meanGradient(Mat & grayImg) {
	if (grayImg.channels() != 1) {
		printf("avgGradient �������󣬱������뵥ͨ��ͼ��");
		return 0.0;
	}
	// ԭ�Ҷ�ͼת���ɸ�������������
	Mat src;
	grayImg.convertTo(src, CV_64FC1);

	double temp = 0.0f;
	// ������һ�ײ�ֵı߽����⣬�������ж�Ҫ-1
	int rows = src.rows - 1;
	int cols = src.cols - 1;

	// ���ݹ�ʽ����ƽ���ݶ�
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			// ��ɢ��delta�������ڵ���ɢ��Ĳ�ֵ
			double dx = src.at<double>(r, c + 1) - src.at<double>(r, c);
			double dy = src.at<double>(r + 1, c) - src.at<double>(r, c);
			double ds = sqrt((dx*dx + dy * dy) / 2);
			temp += ds;
		}
	}
	double imageAVG = temp / (rows*cols);

	return imageAVG;
}

/*����Ҷ�ͼ�ľ�ֵ�ͷ��ʵ���Ϻ��沢û���õ��ú���*/
void mean_std(const Mat & grayImg, double & mean, double & std) {
	if (grayImg.channels() != 1) {
		printf("mean_std �������󣬱������뵥ͨ��ͼ��");
		return;
	}
	Mat mat_mean, mat_stddev;
	meanStdDev(grayImg, mat_mean, mat_stddev);
	mean = mat_mean.at<double>(0, 0);
	std = mat_stddev.at<double>(0, 0);
}

double getMSE(const Mat & src1, const Mat & src2)
{
	//��״��һ��ֱ���˳�����
	if (src1.cols != src2.cols || src1.rows != src2.rows)
	{
		cout << "the sizes of these images are not of the same! " << endl;
		exit(0);
	}
	Mat s1;
	absdiff(src1, src2, s1);    // |src1 - src2|
	s1.convertTo(s1, CV_32F);   // ������8λ��������ƽ������
	s1 = s1.mul(s1);            // |src1 - src2|^2
	Scalar s = sum(s1);         // ����ÿ��ͨ����Ԫ��

	double result = 0.0f;
	int ch = s1.channels();
	for (int i = 0; i < ch; i++) {
		// ��������ͨ��
		result += s.val[i];
	}

	if (result <= 1e-10) // ���ֵ̫С��ֱ�ӵ���0
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
	// �����ɾ��鹫ʽȡ��
	// C1=(K1*L)^2, C2=(K2*L)^2, C3=C2/2, һ���K1=0.01, K2=0.03, L=255�� ������ֵ�Ķ�̬��Χ��һ�㶼ȡΪ255��
	const double C1 = 6.5025, C2 = 58.5225;
	const int TYPE = CV_32F;

	// �����ڵ��ֽ������ϼ��㣬��Χ�������Ҫת��
	Mat I1, I2;
	src1.convertTo(I1, TYPE);
	src2.convertTo(I2, TYPE);
	
	Mat I2_2 = I2.mul(I2);  // I2^2
	Mat I1_2 = I1.mul(I1);  // I1^2
	Mat I1_I2 = I1.mul(I2); // I1*I2

	// ��˹��������ͼ��ľ�ֵ�������Լ�Э��������ǲ��ñ������ص�ķ�ʽ���Ի������ߵ�Ч��
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
	// ��������ͨ����SSIM��ƽ��ֵ��[0,1]֮��
	return (SSIM.val[2] + SSIM.val[1] + SSIM.val[0]) / 3;
}

//Tenengrad�ݶȷ�����Laplacian�ݶȷ���
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
	//ͼ���ƽ���ݶ�
	double meanValue = 0.0;
	meanValue = mean(imageSobel)[0];

	return meanValue;
}

//�����׼�
double meanStdValCount(Mat& imageSource)
{
	Mat imageGrey;

	cvtColor(imageSource, imageGrey, CV_RGB2GRAY);
	Mat meanValueImage;
	Mat meanStdValueImage;

	//��Ҷ�ͼ��ı�׼��
	meanStdDev(imageGrey, meanValueImage, meanStdValueImage);
	double meanValue = 0.0;
	meanValue = meanStdValueImage.at<double>(0, 0);
	return meanValue;
}

/*
*���ܣ���������ʾ��ͼƬ��
*imageSource:Դͼ��meanValue��attention_Str:Ҫ��ʾ�ķ����ͷ������ͣ�x,y,white_width,white_height��ʾ�İ�ɫ�������ʼ�����꣬��ȸ߶�
*/
void putTextOnImg(Mat& imageSource, double meanValue, string attention_Str = "yiqIQA: ", int x = 20, int y = 50, int white_width = 230, int white_height = 30)
{
	Mat newImage(white_height,white_width, CV_8UC3, Scalar(255, 255, 255));//height * width,ɫ���λ��ͨ�������Ϊ��ɫ
	Mat imageROI = imageSource(Rect(x,y-23, white_width,white_height));//x��y��width * height, ����Ϊx������Ϊy����ԭͼ�еĸ���Ȥ����.
	newImage.copyTo(imageROI);

	////Ҳ����ֱ�ӱ���Ԫ�أ���putText�������Ϊ��ɫ
	//for (int i = y-23; i < y+ white_height; i++)  
	//	for (int j =x; j < x+ white_width; j++)
	//	{
	//		imageSource.at<Vec3b>(i, j)[0] = 255; //i ��rows��j��cols,Ҳ���� �к���
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
	double meanStdVal = meanStdValCount(img);  //��׼��
	double gradValue = gradCount(img, "Sobel");  //ƽ���ݶ�
	double entropyVal = entropy(img);   //��Ϣ��
	//double meanGrad = meanGradient(img);
	score = 0.4*meanStdVal+ 0.35*gradValue +0.25*entropyVal ;
	return score;
}

void runDefalut()
{//��ȡ��ǰ��ִ���ļ����ڵ��ļ����ڵ�jpgͼƬ
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
	//PSNR:PSNRֵԽ�󣬱�������ͼ����ο�ͼ��֮���ʧ���С��ͼ�������Ϻá�
	vector<double> psnrScoreVec;
	Mat  referenceImg = imgVec[imgVec.size() - 1];
	for (int i = 0; i < imgVec.size(); i++)
	{
		double mse = getMSE(imgVec[i], referenceImg);
		double score = getPSNR(imgVec[i], referenceImg, mse);
		psnrScoreVec.push_back(score);
		putTextOnImg(imgVec[i], score, "PSNR :", 20, 80);
	}

	//SSIM:SSIMֵԽ��ͼ������Խ��
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
	//TODO:��ʱ�����������ض����ļ����ͣ�������Ĭ�ϸ��ļ�����ֻ��ͼƬ���ͣ�û����������txt
	//�ļ����
	intptr_t hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib & _A_SUBDIR)) { //�Ƚ��ļ������Ƿ����ļ���
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					//�ݹ�����
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0); //Ѱ����һ�����ɹ�����0������-1
		_findclose(hFile);
	}
}

void runByUserPath(string path)
{
	bool isSizeSame(true);
	vector<string> imgPathVec;
	//TODO:��������ֻ��ȡ��·��
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
	//TODO:�����ж�ͼƬ��״�Ƿ�һ��
	int tem_col(imgVec[0].cols), tem_row(imgVec[0].rows);
	for (int i = 1; i < imgVec.size(); i++)
	{
		if (tem_col != imgVec[i].cols || tem_row != imgVec[i].rows)
		{
			isSizeSame = false;
			break;
		}
	}

	int x = 10, y = 10;  //�����ʾ�������Ͻǵ����
	for (int i = 0; i < imgVec.size(); i++)
	{
		if(!isSizeSame) {

			double meanStdVal = meanStdValCount(imgVec[i]);  //��׼��
			double gradValue = gradCount(imgVec[i], "Sobel");  //ƽ���ݶ�
			double entropyVal = entropy(imgVec[i]);   //��Ϣ��
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
		//	imgVec[i].at<Vec3b>(p, q)[0] = 255; //p ��rows��q��cols,Ҳ���� �к���
		//	imgVec[i].at<Vec3b>(p, q)[1] = 255;
		//	imgVec[i].at<Vec3b>(p, q)[2] = 255;
		//     }
		//putText(imgVec[i], imgPathVec[i], Point(0, 30), FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 25, 25), 2);
	}
	//TODO�����Ǹ���ͼƬ�����������������ͼƬ��������������cols,rows,
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
	//imwrite("Img.jpg", img);   //����ͼƬ
	namedWindow("testImg", 0);
	imshow("testImg", img);
	waitKey();
}

int main()
{
	string str;
	cout << "please enter instruction��enter help for help" << endl;
	bool ctrl_flag = true;
	while (cin>>str)
 	{
		    //����������
			if (str == "help" or str =="h")
			{
				cout << "runDefault : Assess the images of the default directory;" << endl;
				cout << "runUserPath��Assess the images of the user's directory" << endl;
				cout << "runSingle:Assess the single image" << endl;
				cout << "cls:clear the console" << endl;
				cout << "quit:exit the console" << endl;
			}
			else if (str == "runDefault" or str =="rd")  
			{//����Ĭ�ϵ�����
				cout << "Default program is running..." << endl;
				runDefalut();
			}
			else if (str == "runUserPath" or str =="ru")
			{//�����û��Զ�������ݣ�Ŀǰ��ʱ��Ϊ����ͼƬ��Сһ��,�����һ�£���ȫ����������ͼƬ�ߴ�ƴ����һ��
				cout << "Please input img path��eg��F://testImg//aligned_Img��:" << endl;
				string imgPath;
				//imgPath = "F://testImg//aligned_Img";
				cin >> imgPath;
				cout << "path:" << imgPath << endl;
				cout << "Program is running..." << endl;
				runByUserPath(imgPath);
			}
			else if (str == "runSingle" or str =="rs")
			{//���Ե���ͼƬ
				cout<<"comming soon!"<<endl;
			 //TODO:runSingle

			}
			else if (str == "cls")
			{ //�������̨
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
