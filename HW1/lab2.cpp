
// FIRST HOMEWORK, COMPUTER VISION - LAB. 1

// GIULIO CONCA 
// ID: 2028844

#include<iostream>
#include<opencv2/opencv.hpp>
#include "filter.h"

//Struct used to pass parameters to the function onChange, called when the slider of a trackbar changes position

struct TrackbarParameters {

  enum class Type {median = 0, gaussian, bilateral};
  
  TrackbarParameters(cv::Mat img, int k, int sigma, Type t) : 
  image(img),
  kernel(k),
  std_dev(sigma),
  filter(t)
  {

  }

  TrackbarParameters(cv::Mat img, int k, Type t) : 
  image(img),
  kernel(k),
  filter(t)
  {
  	
  }

  TrackbarParameters(cv::Mat img, int k, int s_range, int s_space, Type t) : 
  image(img),
  kernel(k),
  sigma_range(s_range),
  sigma_space(s_space),
  filter(t)
  {
  	
  }

  cv::Mat image;
  int kernel;
  int std_dev;
  int sigma_range;
  int sigma_space;
  Type filter;  
};

//Function that was provided by the Professor

void showHistogram(std::vector<cv::Mat>& hists) {
  // Min/Max computation
  double hmax[3] = {0,0,0};
  double min;
  cv::minMaxLoc(hists[0], &min, &hmax[0]);
  cv::minMaxLoc(hists[1], &min, &hmax[1]);
  cv::minMaxLoc(hists[2], &min, &hmax[2]);

  std::string wname[3] = { "blue", "green", "red" };
  cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                           cv::Scalar(0,0,255) };

  std::vector<cv::Mat> canvas(hists.size());

  // Display each histogram in a canvas
  for (int i = 0, end = hists.size(); i < end; i++)
  {
    canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

    for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
    {
      cv::line(
            canvas[i],
            cv::Point(j, rows),
            cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
            hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
            1, 8, 0
            );
    }

    cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
  }
}

//Function that takes an RGB image (passed by copy), computes the histogram and equalize it
//(equalization performed for all the three channels)

void computeAndEqualizeRGB(cv::Mat image) {

  cv::imshow("img", image);
  cv::Mat output;

  std::vector<cv::Mat> bgr_channels(3);
  cv::split(image, bgr_channels);

  int size = 256;
  float range[] = {0,256};
  const float* histRange = {range};
  std::vector<cv::Mat> hist(3);

  calcHist( &bgr_channels[0], 1, 0, cv::Mat(), hist[0], 1, &size, &histRange, true, false );
  calcHist( &bgr_channels[1], 1, 0, cv::Mat(), hist[1], 1, &size, &histRange, true, false );
  calcHist( &bgr_channels[2], 1, 0, cv::Mat(), hist[2], 1, &size, &histRange, true, false );

  showHistogram(hist);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  for(int i = 0; i < 3; ++i) 
      cv::equalizeHist(bgr_channels[i], bgr_channels[i]);

  calcHist( &bgr_channels[0], 1, 0, cv::Mat(), hist[0], 1, &size, &histRange, true, false );
  calcHist( &bgr_channels[1], 1, 0, cv::Mat(), hist[1], 1, &size, &histRange, true, false );
  calcHist( &bgr_channels[2], 1, 0, cv::Mat(), hist[2], 1, &size, &histRange, true, false );

  cv::merge(bgr_channels, output);
  imshow("RGBequalized", output);
  showHistogram(hist);
}

//Function that takes an LAB image (passed by copy), computes the histogram and equalize it
//(equalization performed only for the luminance channel). Then the image is reconverted
//to RGB color space to compute and show the histogram

void computeAndEqualizeLAB(cv::Mat image) {

  cv::Mat output;
  std::vector<cv::Mat> lab_channels(3);
  cv::split(image, lab_channels);

  cv::equalizeHist(lab_channels[0], lab_channels[0]);
  cv::merge(lab_channels, output);

  cv::cvtColor(output, output, cv::COLOR_Lab2BGR);
  cv::waitKey(0);
  cv::destroyAllWindows(); 
  cv::imshow("LABEqualized", output);

  std::vector<cv::Mat> bgr_channels(3);
  cv::split(output, bgr_channels);

  int size = 256;
  float range[] = {0,256};
  const float* histRange = {range};
  std::vector<cv::Mat> hist(3);

  calcHist( &bgr_channels[0], 1, 0, cv::Mat(), hist[0], 1, &size, &histRange, true, false );
  calcHist( &bgr_channels[1], 1, 0, cv::Mat(), hist[1], 1, &size, &histRange, true, false );
  calcHist( &bgr_channels[2], 1, 0, cv::Mat(), hist[2], 1, &size, &histRange, true, false );
  showHistogram(hist);
}

//Function that is called whenever the slider of a trackbar changes its position. Depending on the type of filter
//that must be applied (checked through the parameter "filter" of the object pointed to by the pointer to void), some
//specific operations are performed. An object of the corresponding filter type is created, using the parameters 
//passed to the function (that changes when the slider of a trackbar is moved, as indicated in the createTrackbar
//function).
//In particular, the parameteres that are recovered inside the function represent the values of the trackbars, as 
//indicated in the createTrackbar function. These parameters are used to create an object of the corresponding filter,
//so that the doFilter() function is called every time with the parameters set by the position of the trackbars.

void onChange(int, void* userdata) {
  
  if(((TrackbarParameters*)userdata)->filter == TrackbarParameters::Type::gaussian) {
  	cv::Mat result;
  	cv::Mat img = ((TrackbarParameters*)userdata)->image;
  	int k = ((TrackbarParameters*)userdata)->kernel;
  	int s = ((TrackbarParameters*)userdata)->std_dev;
 	GaussianFilter gaussian(img, k, (float)s/10);
 	gaussian.doFilter();
  	imshow("Gaussian blur", gaussian.getResult());
  }

  if(((TrackbarParameters*)userdata)->filter == TrackbarParameters::Type::median) {
  	cv::Mat result;
  	cv::Mat img = ((TrackbarParameters*)userdata)->image;
  	int k = ((TrackbarParameters*)userdata)->kernel;
 	MedianFilter median(img, k);
 	median.doFilter();
  	imshow("Median blur", median.getResult());
  }

  if(((TrackbarParameters*)userdata)->filter == TrackbarParameters::Type::bilateral) {
  	cv::Mat result;
  	cv::Mat img = ((TrackbarParameters*)userdata)->image;
 	int k = ((TrackbarParameters*)userdata)->kernel;
  	int s_range = ((TrackbarParameters*)userdata)->sigma_range;
  	int s_space = ((TrackbarParameters*)userdata)->sigma_space;
 	BilateralFilter bilateral(img, k, (float)s_range/10, (float)s_space/10);
 	bilateral.doFilter();
  	imshow("Bilateral blur", bilateral.getResult());
  }
}

int main(int argc, char *argv[]) {

  cv::Mat input = cv::imread("overexposed.jpg", cv::IMREAD_COLOR);
  //cv::resize(input, input, cv::Size(input.cols/1.7, input.rows/1.7));
  cv::Mat lab_copy = input.clone();


  // HISTOGRAM COMPUTATION AND EQUALIZATION //


  //RGB equalization
  computeAndEqualizeRGB(input);

  //conversion to LAB color space
  cv::cvtColor(lab_copy, lab_copy, cv::COLOR_BGR2Lab);

  //Luminance equalization
  computeAndEqualizeLAB(lab_copy);


  // FILTERING PART //
  

  //First filter: GAUSSIAN
  TrackbarParameters par_gaussian(input, 1, 1, TrackbarParameters::Type::gaussian);

  cv::waitKey(0);  
  cv::destroyAllWindows();

  //window that contains all the trackbars for the gaussian filter
  cv::namedWindow("Gaussian blur");

  //par_gaussian.kernel contains the value of the kernel, set by the trackbar. This value is recovered
  //inside the function onChange and used to perform the filtering operation (after the creation of an
  //object of the corresponding filter class). Same thing for the other parameters and other filters
  cv::createTrackbar("kernel", "Gaussian blur", &par_gaussian.kernel, 20, onChange, (void*)&par_gaussian);
  cv::createTrackbar("std_dev*10", "Gaussian blur", &par_gaussian.std_dev, 100, onChange, (void*)&par_gaussian);
  imshow("Gaussian blur", input);

  cv::waitKey(0);  
  cv::destroyAllWindows();

  //Second filter: MEDIAN
  TrackbarParameters par_median(input, 1, TrackbarParameters::Type::median);

  //window that contains all the trackbars for the median filter
  cv::namedWindow("Median blur");

  cv::createTrackbar("kernel", "Median blur", &par_median.kernel, 20, onChange, (void*)&par_median);
  imshow("Median blur", input);

  cv::waitKey(0);  
  cv::destroyAllWindows();

  //Third filter: BILATERAL
  TrackbarParameters par_bilateral(input, 1, 1, 1, TrackbarParameters::Type::bilateral);

  //window that contains all the trackbars for the bilateral filter
  cv::namedWindow("Bilateral blur");

  cv::createTrackbar("kernel", "Bilateral blur", &par_bilateral.kernel, 30, onChange, (void*)&par_bilateral);
  cv::createTrackbar("sigma_range*10", "Bilateral blur", &par_bilateral.sigma_range, 3000, onChange, (void*)&par_bilateral);
  cv::createTrackbar("sigma_space*10", "Bilateral blur", &par_bilateral.sigma_space, 600, onChange, (void*)&par_bilateral);
  imshow("Bilateral blur", input);

  cv::waitKey(0);  
  cv::destroyAllWindows();

}