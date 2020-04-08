#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#define Z_MIN 0
#define Z_MAX 255

using namespace cv;
using namespace std;

int offsets[][2] = {
  {-1, 1}, {0, 1}, {1, 1} ,
  {-1, 0}, {0, 0}, {1, 0} ,
  {-1,-1}, {0,-1}, {1,-1} };

void loadExposureSeq(String, vector<Mat>&, vector<float>&);
double median(Mat image);
uchar grayMedian(Mat& gray);
template <class T>
void offsetImage(Mat& input, int xoffset, int yoffset, Mat& output);
void MTB(vector<Mat>& images, int level_num = 5);
void reconstructRadianMap(Mat& radian_map,
    vector<Mat>& images, vector<float>& times, double lambda);
void photographicToneMapping(Mat& Ld, const Mat Lw, const float a = 0.18f, const float phi = 3);


int main(int argc, char** argv)
{

    //CommandLineParser parser(argc, argv, "{@input | D:/maochinn/Lib/opencv_extra/testdata/cv/hdr/exposures | Input directory that contains images and exposure times. }");
	CommandLineParser parser(argc, argv, "{@input | D:/maochinn/NTU/VFX/images/2 | Input directory that contains images and exposure times. }");

    //! [Load images and exposure times]
    vector<Mat> images;
    vector<float> times;
    loadExposureSeq(parser.get<String>("@input"), images, times);
    //! [Load images and exposure times]

	//for (auto& img : images)
	//{
	//	pyrDown(img, img);
	//	pyrDown(img, img);
	//}

	if (images.empty())
	{
		puts("load image error!");
		return 0;
	}

	MTB(images);
    Mat hdr(images[0].rows, images[0].cols, CV_32FC3);
    reconstructRadianMap(hdr, images, times, 15.0);

	imwrite("output.hdr", hdr);

	for (int i = 0;i<images.size();i++)
	{
		imwrite(to_string(i) + "_alignment.png", images[i]);
	}

    //Mat hdr = imread("output.hdr", -1);
    Mat ldr(hdr.rows, hdr.cols, CV_8UC3);

    imwrite("output.hdr", hdr);
    photographicToneMapping(ldr, hdr, 0.18f);
    imshow("tonemap 0.18", ldr);
	imwrite("tonemap_18.png", ldr);
    photographicToneMapping(ldr, hdr, 0.36f);
    imshow("tonemap 0.36", ldr);
	imwrite("tonemap_36.png", ldr);
    photographicToneMapping(ldr, hdr, 0.5f);
    imshow("tonemap 0.5", ldr);
	imwrite("tonemap_50.png", ldr);
    photographicToneMapping(ldr, hdr, 0.7f);
    imshow("tonemap 0.7", ldr);
	imwrite("tonemap_70.png", ldr);
    photographicToneMapping(ldr, hdr, 0.9f);
    imshow("tonemap 0.9", ldr);
	imwrite("tonemap_90.png", ldr);

    waitKey(0);
    destroyAllWindows();

    return 0;
}

void loadExposureSeq(String path, vector<Mat>& images, vector<float>& times)
{
    path = path + "/";
    ifstream list_file((path + "list.txt").c_str());
    string name;
    float val;
    while (list_file >> name >> val) {
        Mat img = imread(path + name);
        images.push_back(img);
        times.push_back(1 / val);
    }
    list_file.close();
}

double median(Mat channel)
{
    double m = (channel.rows * channel.cols) / 2;
    int bin = 0;
    double med = -1.0;

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    for (int i = 0; i < histSize && med < 0.0; ++i)
    {
        bin += cvRound(hist.at< float >(i));
        if (bin > m&& med < 0.0)
            med = i;
    }
    
    return med;
}
uchar grayMedian(Mat& gray)
{
    std::vector<uchar> pixels;
    for (int i = 0; i < gray.rows; i++)
        for (int j = 0; j < gray.cols; j++)
        {
            pixels.push_back(gray.at<uchar>(i, j));
        }
    sort(pixels.begin(), pixels.end());
    return pixels[pixels.size() / 2];
}

template <class T>
void offsetImage(Mat& input, int xoffset, int yoffset, Mat& output)
{
    // https://stackoverflow.com/questions/17145263/whats-the-best-way-to-offset-an-image-using-opencv
    //Mat temp(input.rows + 2 * yoffset, input.cols + 2 * xoffset, input.type());
    //Mat roi(temp(cvRect(xoffset, yoffset, input.cols, input.rows)));
    //input.copyTo(roi);
    //output = temp.clone();

    for (int i = 0; i < output.rows; i++)
        for (int j = 0; j < output.cols; j++)
        {
            int idx_x = i - xoffset;
            int idx_y = j - yoffset;
            idx_x = max(0, min(idx_x, input.rows-1));
            idx_y = max(0, min(idx_y, input.cols-1));
            output.at<T>(i, j) = input.at<T>(idx_x, idx_y);
        }
}


void MTB(vector<Mat>& images, int level_num)
{
    // refer: https://ssarcandy.tw/2017/04/16/High-Dynamic-Range-Imaging/

    vector<vector<Mat>> images_levels;
    vector<vector<Mat>> maskes_levels;

    int src_idx = images.size() / 2;
    //int src_idx = 6;

    for (int i = 0;i<images.size();i++)
    {
        Mat gray;
        cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
        int median = grayMedian(gray);
        std::vector<Mat> levels;
        std::vector<Mat> maskes;
        for (int j = 0; j < level_num; j++)
        {
            Mat half_img;
			if (levels.empty())
				gray.copyTo(half_img);
            else
                pyrDown(levels.back(), half_img);
            levels.push_back(half_img);
        }
        for (Mat& lv : levels)
        {
            Mat mask;
            inRange(lv, median - 5, median + 5, mask);
            mask = ~mask;
            maskes.push_back(mask);

            cv::threshold(lv, lv, median, 255, cv::THRESH_BINARY);
        } 
        images_levels.push_back(levels);
        maskes_levels.push_back(maskes);
    }

    for (int i = 0;i< images_levels.size();i++)
    {
        if (i == src_idx)
            continue;
        
        int offset_x = 0;
        int offset_y = 0;

        for (int j = level_num-1; j >= 0; j--)
        {
            //first image is source
            Mat& src = images_levels[src_idx][j];
            //Mat& mask = maskes_levels[src_idx][j];
            std::vector<int> diff;

            offset_x *= 2;
            offset_y *= 2;

            for (int k = 0; k < 9; k++)
            {
				Mat dst_mask;
				maskes_levels[i][j].copyTo(dst_mask);
				offsetImage<uchar>(maskes_levels[i][j], offset_x + offsets[k][0], offset_y + offsets[k][1], dst_mask);
                Mat mask = dst_mask & maskes_levels[src_idx][j];
                Mat dst;
                src.copyTo(dst);
                offsetImage<uchar>(images_levels[i][j], offset_x + offsets[k][0], offset_y + offsets[k][1], dst);
                diff.push_back(cv::sum((src ^ dst) & mask)[0]);
            }
            int idx = 4;
            int diff_min = diff[idx];
            for (int d = 0; d < diff.size(); d++)
            {
                if (diff[d] < diff_min)
                {
                    idx = d;
                    diff_min = diff[d];
                }
            }

            offset_x += offsets[idx][0];
            offset_y += offsets[idx][1];
        }
        //
        Mat temp;
        images[i].copyTo(temp);
        offsetImage<Vec3b>(temp, offset_x, offset_y, images[i]);

		printf("image %d offset (%d, %d)\n", i, offset_x, offset_y);
    }
}

int w(int z)
{
    return z <= (Z_MIN + Z_MAX) / 2 ? z - Z_MIN : Z_MAX - z;
}

void reconstructRadianMap(Mat& radian_map,
    vector<Mat>& ldrs, vector<float>& times, double lambda)
{
    vector<Mat> images;
    for (Mat ldr : ldrs)
    {
        images.push_back(ldr);
    }

    int n = 256;
    int P = images.size();
    int N = (Z_MAX - Z_MIN) / (P - 1);
    
	//N = 200;

    while(1)
    {
        if ((images[0].rows * images[0].cols) / 16 < N)
            break;
        for(Mat& img : images)
        {
            pyrDown(img, img);
        }
    }
    N = images[0].rows * images[0].cols;

    vector<float> B;
    for (float time : times)
    {
        B.push_back(log(time));
    }
    for (int channel = 0; channel < 3; channel++)
    {
        vector<vector<int>> Z;
        for (Mat& img : images)
        {
            vector<int> pixels;
            for (int x = 0; x < img.cols; x++)
                for (int y = 0; y < img.rows; y++)
                {
                    pixels.push_back(img.at<Vec3b>(y, x)[channel]);
                }
            Z.push_back(pixels);
        }

		//for (Mat& img : images)
		//{
		//	vector<int> pixels;
		//	Z.push_back(pixels);
		//}
		//for (int i = 0; i < N; i++)
		//{
		//	int x = rand() % images[0].cols;
		//	int y = rand() % images[0].rows;
		//	
		//	for (int j = 0; j < Z.size(); j++)
		//	{
		//		Z[j].push_back(images[j].at<Vec3b>(y, x)[channel]);
		//	}
		//}



        Mat A = Mat::zeros(cv::Size(256 + N, N * P + 1 + (n-2)), CV_64FC1);
        Mat b = Mat::zeros(cv::Size(1, N * P + 1 + (n-2)), CV_64FC1);
        int k = 0;
        for(int j = 0;j< Z.size();j++)
            for (int i = 0; i < Z[j].size(); i++)
            {
                int wij = w(Z[j][i]);
                A.at<double>(k, Z[j][i]) = wij;
                A.at<double>(k, n + i) = -wij;
                b.at<double>(k, 0) = wij * B[j];
                k++;
            }
        A.at<double>(k++, 127) = 1.0;
        for (int i = 0; i < n - 2; i++)
        {
            A.at<double>(k, i) = lambda * w(i + 1);
            A.at<double>(k, i+1) = -2.0 * lambda * w(i + 1);
            A.at<double>(k, i+2) = lambda * w(i + 1);
            k++;
        }

        Mat A_inversed;
        invert(A, A_inversed, DECOMP_SVD);
        Mat x = A_inversed * b;

        vector<double> g, lnE;
        for (int i = 0; i < n; i++)
        {
            g.push_back(x.at<double>(i, 0));
        }
        for (int i = n; i < x.rows; i++)
        {
            lnE.push_back(x.at<double>(i, 0));
        }

        for (int x = 0; x < radian_map.cols; x++)
            for (int y = 0; y < radian_map.rows; y++)
            {
                float lnEi = 0.0f;
                int w_sum = 0;
                for (int j = 0; j < P; j++)
                {
                    int zij = ldrs[j].at<Vec3b>(y, x)[channel];
                    lnEi += w(zij) * (g[zij] - B[j]);
                    w_sum += w(zij);
                }
                lnEi /= (float)w_sum;
                radian_map.at<Vec3f>(y, x)[channel] = exp(lnEi);
                //Vec3f temp = radian_map.at<Vec3f>(y, x);
                //puts("");
            }

        
        
    }
    
}

float length(Vec3f v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void photographicToneMapping(Mat& ldr, const Mat hdr, const float a, const float phi)
{
	Mat Lw(hdr.rows, hdr.cols, CV_32FC1);
    int N = Lw.cols * Lw.rows;

	for (int x = 0; x < Lw.cols; x++)
		for (int y = 0; y < Lw.rows; y++)
		{
			Vec3f rgb = hdr.at<Vec3f>(y, x);
			//Lw.at<float>(y, x) = 0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2];
			Lw.at<float>(y, x) = 0.299f * rgb[2] + 0.587f * rgb[1] + 0.114f * rgb[0];
		}

	float Lw_average = 0.0f;
    Mat log_Lw = Mat::zeros(Size(Lw.cols, Lw.rows), CV_32FC1);
    log(Lw + 0.0001f, log_Lw);
    for (int x = 0; x < Lw.cols; x++)
        for (int y = 0; y < Lw.rows; y++)
        {
            Lw_average += log_Lw.at<float>(y, x);
        }
	Lw_average = exp(Lw_average / N);

	Mat Lm(Lw.rows, Lw.cols, CV_32FC1);
	Mat Ld(Lw.rows, Lw.cols, CV_32FC1);
	for (int x = 0; x < Lm.cols; x++)
		for (int y = 0; y < Lm.rows; y++)
		{
			Lm.at<float>(y, x) = (a / Lw_average) * Lw.at<float>(y, x);

			//for global operator
			//Ld.at<float>(y, x) = Lm.at<float>(y, x) / (1.0f + Lm.at<float>(y, x));
		}

	//for local operator

	vector<Mat> Lblur;
	for (int i = 1; i <= 21; i+=2)
	{
		Mat blur;
		GaussianBlur(Lm, blur, Size(i, i), 0.0f, 0.0f);
		Lblur.push_back(blur);
	}

	vector<Mat> V;
	for (int s = 0; s + 1 < Lblur.size(); s++)
	{
		Mat v = Lblur[s] - Lblur[s + 1];
		for (int x = 0; x < v.cols; x++)
			for (int y = 0; y < v.rows; y++)
			{
				v.at<float>(y, x) /= pow(2, phi) * (a / ((s + 1) * (s + 1))) + Lblur[s].at<float>(y, x);
			}
		V.push_back(v);
	}

	for (int x = 0; x < Ld.cols; x++)
		for (int y = 0; y < Ld.rows; y++)
		{
			int s_max = 0;
			for (int s = 0; s < V.size(); s++)
			{
				float temp = abs(V[s].at<float>(y, x));
				if (abs(V[s].at<float>(y, x)) < 0.001f)
					s_max = s;
				else
					break;
			}

			Ld.at<float>(y, x) = Lm.at<float>(y, x) / (1.0f + Lblur[s_max].at<float>(y, x));
		}

	//get back to rgb 
	for (int x = 0; x < ldr.cols; x++)
		for (int y = 0; y < ldr.rows; y++)
		{
			ldr.at<Vec3b>(y, x) = 255 * Ld.at<float>(y, x) * hdr.at<Vec3f>(y, x) / (Lw.at<float>(y, x) + 0.0001f);
		}

}