#include <iostream>
#include <opencv2/opencv.hpp>




int main()
{
	//std::cout << "a wen";
	cv::Mat img = cv::imread("C:\\Users\\DELL\\source\\repos\\Project1\\bus.jpg", 1);
	cv::imshow("bus", img);
	cv::waitKey(0);
	std::cout << "a wen"<<std::endl;
	return 0;
}