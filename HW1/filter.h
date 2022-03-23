// Computer vision 2021 (P. Zanuttigh) - LAB 2
#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Generic class implementing a filter with the input and output image data and the parameters
class Filter{

// Methods

public:

	// constructor 
	// input_img: image to be filtered
	// filter_size : size of the kernel/window of the filter
	Filter(cv::Mat input_img, int size);

	// perform filtering (in base class do nothing, to be reimplemented in the derived filters)
	virtual void doFilter();

	// get the output of the filter
	cv::Mat getResult();

	//set the window size (square window of dimensions size x size)
	void setSize(int size);
	
	//get the Window Size
	int getSize() const;

// Data

protected:

	// input image
	cv::Mat input_image;

	// output image (filter result)
	cv::Mat result_image;

	// window size
	int filter_size;

};

// Gaussian Filter
class GaussianFilter : public Filter  {

public:

	GaussianFilter(cv::Mat input_img, int size, float standard_dev);

	void doFilter() override;

private:

	float std_dev;

// place constructor
// re-implement  doFilter()
// additional parameter: standard deviation (sigma)

};

class MedianFilter : public Filter {

public:

	MedianFilter(cv::Mat input_img, int size);

	void doFilter() override;

};

class BilateralFilter : public Filter {

public:

	BilateralFilter(cv::Mat input_img, int size, float sigma_r, float sigma_s);

	void doFilter() override;

private:

	float sigma_range;
	float sigma_space;


// place constructor
// re-implement  doFilter()
// additional parameters: sigma_range, sigma_space


};

#endif