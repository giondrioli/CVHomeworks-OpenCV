// Computer Vision 2021 (P. Zanuttigh) - LAB2 

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "filter.h"

//using namespace cv;

	// constructor
	Filter::Filter(cv::Mat input_img, int size) {

		input_image = input_img;
		if (size % 2 == 0)
			size++;
		filter_size = size;
	}

	// for base class do nothing (in derived classes it performs the corresponding filter)
	void Filter::doFilter() {

		// it just returns a copy of the input image
		result_image = input_image.clone();

	}

	// get output of the filter
	cv::Mat Filter::getResult() {

		return result_image;
	}

	//set window size (it needs to be odd)
	void Filter::setSize(int size) {

		if (size % 2 == 0)
			size++;
		filter_size = size;
	}

	//get window size 
	int Filter::getSize() const {

		return filter_size;
	}



	GaussianFilter::GaussianFilter(cv::Mat input_img, int size, float standard_dev) : 
	Filter(input_img, size),
	std_dev(standard_dev)
	{

	}

	void GaussianFilter::doFilter() {
		cv::GaussianBlur(input_image, result_image, cv::Size(filter_size, filter_size), std_dev, 0);
	}


	MedianFilter::MedianFilter(cv::Mat input_img, int size) :
	Filter(input_img, size)
	{

	} 

	void MedianFilter::doFilter() 
	{
		cv::medianBlur(input_image, result_image, filter_size);
	}

	BilateralFilter::BilateralFilter(cv::Mat input_img, int size, float sigma_r, float sigma_s) :
	Filter(input_img, size),
	sigma_range(sigma_r),
	sigma_space(sigma_s)
	{

	}

	void BilateralFilter::doFilter() 
	{
		cv::bilateralFilter(input_image, result_image, filter_size, sigma_range, sigma_space);
	}
