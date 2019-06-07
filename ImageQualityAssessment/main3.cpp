
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


//Mat combineImages(vector<Mat>imgs,//@parameter1:��Ҫ��ʾ��ͼ���� 
//	int col,//parameter2:��ʾ������
//	int row, //parameter3:��ʾ������
//	bool hasMargin) //parameter4:�Ƿ����ñ߿�
//{
//	int imgAmount = imgs.size();//��ȡ��Ҫ��ʾ��ͼ������
//	int width = imgs[0].cols;//������Ĭ����Ҫ��ʾ��ͼ���С��ͬ
//	int height = imgs[0].rows;//��ȡͼ����
//	int newWidth, newHeight;//��ͼ����
//	if (!hasMargin) {
//		newWidth = col * imgs[0].cols;//�ޱ߿���ͼ���/��=ԭͼ���/��*��/����
//		newHeight = row * imgs[0].rows;
//	}
//	else {
//		newWidth = (col + 1) * 20 + col * width;//�б߿�Ҫ���ϱ߿�ĳߴ磬�������ñ߿�Ϊ20px
//		newHeight = (row + 1) * 20 + row * height;
//	}
//
//	Mat newImage(newHeight, newWidth, CV_8UC3, Scalar(255, 255, 255));//��ʾ�����趨�ߴ���µĴ�ͼ��ɫ���λ��ͨ�������Ϊ��ɫ
//
//	int x, y, imgCount;//x�кţ�y�кţ�imgCountͼƬ���
//	if (hasMargin) {//�б߿�
//		imgCount = 0;
//		x = 0; y = 0;
//		while (imgCount < imgAmount) {
//			Mat imageROI = newImage(Rect(x*width + (x + 1) * 20, y*height + (y + 1) * 20, width, height));//��������Ȥ����
//			imgs[imgCount].copyTo(imageROI);//��ͼ���Ƶ���ͼ��
//			imgCount++;
//			if (x == (col - 1)) {
//				x = 0;
//				y++;
//			}
//			else {
//				x++;
//			}//�ƶ����кŵ���һ��λ��
//		}
//	}
//	else {//�ޱ߿�
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
//	return newImage;//�����µ����ͼ��
//};


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

/*����Ҷ�ͼ�ľ�ֵ�ͷ���*/
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
	//��״��һ��ֱ���˳�
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
		return 10000;
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
	//Mat imageSource = imread("F:\\������4.jpg");
	//Mat imageSource = imread("F://lenna.bmp");
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
	//PSNR,PSNRֵԽ�󣬱�������ͼ����ο�ͼ��֮���ʧ���С��ͼ�������Ϻá�
	vector<double> psnrScoreVec;
	Mat  referenceImg = imgVec[imgVec.size() - 1];
	for (int i = 0; i < imgVec.size(); i++)
	{
		double mse = getMSE(imgVec[i], referenceImg);
		double score = getPSNR(imgVec[i], referenceImg, mse);
		psnrScoreVec.push_back(score);
		putTextOnImg(imgVec[i], score, "PSNR score:", 20, 80);
	}

	//SSIM:SSIMֵԽ��ͼ������Խ��
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

	//string *imgfilePath = new string[n];
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
	//TODO��������θ���ͼƬ�������Զ������������ͼƬ��������������cols,rows,
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
				cout << "Default running..." << endl;
				runDefalut();
			}
			else if (str == "runUserPath" or str =="ru")
			{//�����û��Զ�������ݣ�Ŀǰ��ʱ��Ϊ����ͼƬ��Сһ��
				cout << "plz insert img path��eg:F://testImg��:" << endl;
				string imgPath;
				imgPath = "F:\\testImg2";
				//cin >> imgPath;
				cout << "path:" << imgPath << endl;
				//string isSizeSame;
				//bool isSame;
				//cout << "Are the sizes of the images the same? true or false" << endl;
				//cin >> isSizeSame;  //bool�����޷�ͨ��cinֱ������
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
