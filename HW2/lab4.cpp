//FINAL PROJECT - COMPUTER VISION - PANORAMIC IMAGES
//NAME: GIULIO
//SURNAME: CONCA
//ID: 2028844


//WITH THIS SETTING, THE PROGRAM WORKS FOR THE KITCHEN DATASET. 

//TO CHANGE THE SET OF IMAGES, SOME PARAMETERS MUST BE MODIFIED:
//1) THE STRINGS "NAME" INSIDE THE LOADANDPROJECT FUNCTION (ALSO THE EXTENSION)
//2) THE NUMBER OF IMAGES INSIDE THE "NUM" VARIABLE IN THE MAIN 
//3) HALF OF THE FIELD OF VIEW INSIDE THE VARIABLE "ANGLE" IN THE MAIN (27 FOR THE DOLOMITES DATASET, 33 FOR THE OTHERS)
//4) OF COURSE THE SET OF IMAGES MUST BE PLACED INSIDE THE FOLDER, SINCE NOW ONLY THE KITCHEN IMAGES ARE THERE

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "panoramic_utils.h"
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <cmath>
#include <random>

/*
THE FUNCTION LOADS THE IMAGES THAT MUST BE MERGED TOGETHER, AND PROJECT THEM ON A CYLINDRICAL SURFACE.
THEY ARE FIRST CONVERTED TO LAB COLOR SPACE, AND THEN STORED IN A VECTOR CALLED "IMAGES".
THEN, EACH IMAGE IS SPLIT IN ITS THREE CHANNELS, THE LUMINANCE ONE IS EQUALIZED AND THEY
ARE ALL PLACED IN A VECTOR OF VECTORS OF IMAGES CALLED "ALL_CHANNELS", FROM WHICH THEY ARE PROJECTED ON A 
CYLINDRICAL SURFACE. EACH POSITION OF ALL_CHANNELS WILL CONTAIN THE THREE PLANES OF AN IMAGE OF THE SET.
THE VECTOR OF VECTORS IS THEN RETURNED BY THE FUNCTION.
*/
std::vector<std::vector<cv::Mat>> loadAndProject(int n_images, int fov) {
	std::vector<cv::Mat> images;
  	std::vector<cv::Mat> channels(3); 

  	std::string name;
  	for(int i = 1; i < n_images+1; ++i) {
		if(i < 10) 
			name = std::string("i") + std::to_string(0) + std::to_string(i) + std::string(".bmp");
		else
			name = std::string("i") + std::to_string(i) + std::string(".bmp");
		cv::Mat to_add = cv::imread(name);
		cv::cvtColor(to_add, to_add, cv::COLOR_BGR2Lab);//
		images.push_back(to_add);
	}

	std::vector<std::vector<cv::Mat>> all_channels(images.size(), std::vector<cv::Mat>(3));
	for(int i = 0; i < images.size(); ++i) {
		cv::split(images[i], channels);
  		cv::equalizeHist(channels[0], channels[0]);//
		for(int j = 0; j < 3; ++j) {
  			all_channels[i][j] = channels[j];
  		}
  		channels.clear();
	}
	PanoramicUtils cylinder{};
	for(int i = 0; i < all_channels.size(); ++i) {
		for(int j = 0; j < all_channels[i].size(); ++j) {
		all_channels[i][j] = cylinder.cylindricalProj(all_channels[i][j], fov);
		}
	}
	return all_channels;
}

/*
THE FUNCTION CREATES THE TWO MASKS USED BY THE DETECTANDCOMPUTE METHOD. THE MASK NUMBER ONE, THAT WILL BE
USED FOR THE FIRST IMAGE OF THE COUPLE, HIGHLIGHTS THE THREE QUARTERS OF THE ORIGINAL IMAGE ON THE RIGHT HAND SIDE,
WHILE THE MASK THAT WILL BE USED FOR THE SECOND IMAGE OF THE COUPLE HIGHLIGHTS THE THREE QUARTERS OF THE ORIGINAL
IMAGE ON THE LEFT HAND SIDE (IMAGES OF THEM IN THE REPORT). THE AREA THAT SHOULD BE CONSIDERED BY THE MASKS MUST 
BE SET TO 255, WHILE THE PART OF THE IMAGE THAT SHOULD NOT BE CONSIDERED MUST BE SET TO ZERO.
THE TWO MASKS ARE STORED IN A VECTOR OF IMAGES, THAT IS RETURNED BY THE FUNCTION.
*/
std::vector<cv::Mat> maskCreation(cv::Mat image) {
	cv::Mat mask1 = cv::Mat::zeros(image.size(), CV_8U);
	cv::Mat area1(mask1, cv::Rect(image.size().width/4, 0, 0.75*(image.size().width), image.size().height));
	area1 = cv::Scalar(255);

	cv::Mat mask2 = cv::Mat::zeros(image.size(), CV_8U);
	cv::Mat area2(mask2, cv::Rect(0, 0, 0.75*(image.size().width), image.size().height));
	area2 = cv::Scalar(255);
	std::vector<cv::Mat> mask;
	mask.push_back(mask1);
	mask.push_back(mask2);
	return mask;
}

/*
FUNCTION THAT IS USED TO CHECK THE ORDER OF IMAGES. IF THE FIRST IMAGES OF THE SET HAVE BEEN TAKEN ROTATING THE 
CAMERA FROM RIGHT TO LEFT, THEY MUST BE REORDERED (SEE THE REPORT FOR MORE DETAILS). THE FUNCTION LOOKS FOR THE 
FIRST POSITIVE HORIZONTAL TRANSITION AND STORE ITS INDEX IN THE "FIRST_POSITIVE" VARIABLE. THAT INDEX REPRESENTS
ALSO THE INDEX OF THE IMAGE THAT MUST BE PLACED IN THE FIRST POSITION IN THE NEW ORDER (TO HAVE ONLY CLOCKWISE 
ROTATIONS OF THE CAMERA). IN CASE IMAGES MUST BE REORDERED, THE FLAG "OUT_OF_ORDER" IS SET TO ONE. 
IN THIS CASE, THE IMAGE IN POSITION "FIRST POSITIVE" (INSIDE THE VECTOR OF VECTORS OF IMAGES CALLED ALL_CHANNELS) 
IS SWAPPED WITH THE IMAGE IN THE FIRST POSITION. AFTER THAT, THE ORDER OF IMAGES INSIDE ALL_CHANNELS IS CORRECT,
AND THE VALUE OF OUT_OF_ORDER IS RETURNED BY THE FUNCTION (1 IF THEY HAVE BEEN REORDERED, 0 IF THE ORDER WAS CORRECT).
*/
int wrongOrder(std::vector<std::vector<float>> set_translations, std::vector<std::vector<cv::Mat>>& all_channels) {
	int first_positive = -1;
	int out_of_order = 0;
	for(int i = 0; i < set_translations.size(); i++) {
		if(set_translations[i][0] > 0) {
			first_positive = i;
			i = set_translations.size() - 1; 
		}
		else {
			out_of_order = 1;
		}
	}

	std::vector<cv::Mat> temp_im(3);

	if(out_of_order) {
		for(int i = 0; i < 3; ++i) {
			temp_im[i] = all_channels[0][i];
		}

		for(int i = 0; i < 3; i++) {
			all_channels[0][i] = all_channels[first_positive][i];
			all_channels[first_positive][i] = temp_im[i];
		}
	}
	return out_of_order;
}

/*
FUNCTION THAT IMPLEMENTS A SIMPLE VERSION OF THE RANSAC ROBUST ESTIMATOR (DETAILS ON THE REPORT).
FIRST IT SELECTS A RANDOM MATCH INSIDE THE VECTOR OF MATCHES BY GENERATING A RANDOM NUMBER, THENIT COMPUTES THE 
TRANSLATION FOR THAT SPECIFIC RANDOM MATCH. IT ALSO COUNTS HOW MANY MATCHES INSIDE THE VECTOR OF MATCHES AGREE WITH
THAT TRANSLATION. THESE OPERATIONS ARE REPEATED 50 TIMES. WHEN IT FINDS A NUMBER OF COMPATIBLE MATCHES THAT IS 
GREATER THAN THE PREVIOUS NUMBER OF COMPATIBLE MATCHES, THE TRANSLATION RELATED TO THOSE MATCHES ARE STORED IN THE
VECTOR "VEC_TRANSLATIONS". AT THE END OF THE FOR LOOP, THE FINAL TRANSLATION IS COMPUTED BY TAKING THE AVERAGE OF THE
TRANSLATIONS INSIDE VEC_TRANSLATIONS. THE RANSAC FUNCTION IS CALLED FOR EACH COUPLE OF IMAGES IN ORDER TO ESTIMATE
THE TRANSLATION NEEDED TO MERGE THEM TOGETHER. THE FINAL TRANSLATION IS RETURNED BY THE FUNCTION.
*/
std::vector<float> ransac(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, cv::Mat image) {
	//Generations of random numbers between 0 and the size of the vector that contains the matches
	std::random_device random;
	std::mt19937 generator(random());
	std::uniform_int_distribution<> dist(0,matches.size()-1);

	std::vector<cv::Point2f> temp_trans{};			//vector that will contain the translations for all the matches that agree with a random match
	std::vector<cv::Point2f> vec_translations{};	//vector that will contain the final translations for the matches 
	int max = 0;

	for(int i = 0; i < 50; ++i) {
		int rand = dist(generator);
		float x1 = kp1[matches[rand].queryIdx].pt.x;
		float x2 = kp2[matches[rand].trainIdx].pt.x;
		float y1 = kp1[matches[rand].queryIdx].pt.y;
		float y2 = kp2[matches[rand].trainIdx].pt.y;
		float deltax = x2 + (image.size().width - x1);
		float deltay = y2 - y1;
		for(int j = 0; j < matches.size(); ++j) {
			float xj1 = kp1[matches[j].queryIdx].pt.x;
			float xj2 = kp2[matches[j].trainIdx].pt.x;
			float yj1 = kp1[matches[j].queryIdx].pt.y;
			float yj2 = kp2[matches[j].trainIdx].pt.y;
			float deltaxj = xj2 + (image.size().width - xj1);
			float deltayj = yj2 - yj1;
			if((abs(deltaxj - deltax) + abs(deltayj - deltay)) < 5) {
				temp_trans.push_back(cv::Point2f(deltaxj, deltayj));
			}
		}
		if(temp_trans.size() > max) {
			max = temp_trans.size();
			vec_translations = temp_trans;
			temp_trans.clear();
		}
		else {
			temp_trans.clear();
		}
	}

	float xc = 0;
	float yc = 0;

	for(int i = 0; i < vec_translations.size(); i++) {
		xc = xc + vec_translations[i].x;
		yc = yc + vec_translations[i].y;
	}
		
	float final_x = xc/vec_translations.size();		
	float final_y = yc/vec_translations.size();		

	if(final_x > image.size().width) {
		final_x = image.size().width - final_x;
	}
		
	std::vector<float> single_translation;
	single_translation.push_back(final_x);
	single_translation.push_back(final_y);
	return single_translation;
}

/*
FUNCTION THAT PERFORMS THE FEATURE DETECTION, EXTRACTION AND MATCHING BETWEEN COUPLES OF ADJACENT IMAGES.
SINCE THE STITCHED IMAGES WILL BE STORED INSIDE A VECTOR CALLED PANORAMA, THE FIRST POSITION OF THAT VECTOR
IS OCCUPIED BY THE FIRST IMAGE OF THE SET, THAT IS RETRIEVED BY MERGING ITS CHANNELS AND BY RECONVERTING IT 
TO THE RGB COLOR SPACE.
THEN, WITH A FOR LOOP, ONLY THE LUMINANCE PLANE OF THE IMAGES OF EACH COUPLE IS COMSIDERED TO COMPUTE THE TRANSLATION
NEEDED TO MERGE THE IMAGES, BECAUSE THE OTHER PLANES WILL BE TRANSLATED OF THE SAME AMOUNT.
THE FUNCTION USES EITHER SIFT OR ORB, DEPENDING ON THE CONTENT OF THE STRING THAT IS PASSED TO THE FUNCTION.
FOR EACH IMAGE OF THE COUPLE, THE METHODS DETECTANDCOMPUTE (THAT USES ALSO THE MASKS COMPUTED PREVIOUSLY) AND 
THE METHOD MATCH ARE CALLED, IN ORDER TO MATCH THE CORRESPONDING FEATURE. A REFINEMENT OF THE MATCHES IS PERFORMED
BY CHECKING THE DISTANCE OF EACH MATCH AND BY COMPARING IT TO THE MINIMUM DISTANCE AMONG THEM.
THE TRANSLATION FOR THAT SPECIFIC COUPLE OF IMAGES IS COMPUTED USING THE RANSAC FUNCTION, AND EACH TRANSLATION
IS STORED INSIDE THE VECTOR "SET_TRANSLATIONS", THAT AT THE END WILL CONTAIN ALL THE TRANSLATIONS NEEDED TO MERGE
ALL THE IMAGES TOGETHER. 
THEN, THE ORDER OF THE IMAGES IS CHECKED BY CALLING THE WRONGORDER FUNCTION, THAT WILL REORDER THEM, IF NEEDED.
IN THIS CASE, THE MATCHING FUNCTION IS CALLED AGAIN TO EXECUTE THE OPERATIONS ALREADY DESCRIPTED WITH A CORRECT
ORDER OF THE IMAGES.
IF NO REORDER IS NEEDED, THE FUNCTION DEFINE A TRANSLATION MATRIX USING THE VALUES STORED INSIDE SET_TRANSLATIONS.
THE THREE PLANES OF THE SECOND IMAGE OF THE COUPLE ARE TRANSLATED ACCORDINGLY USING THE FUNCTION WARPAFFINE, 
THEN THEY ARE MERGED TOGETHER AND RECONVERTED TO RGB COLOR SPACE. THEN THE "BLUE" PART, THAT IS THE RESULT OF THE
TRANSLATION, IS ELIMINATED AND FINALLY THE FIRST AND THE SECOND IMAGE OF THE COUPLE (THAT IS TRANSLATED) ARE MERGED
TOGETHER BY USING THE FUNCTION HCONCAT. THE RESULTING IMAGE IS INSERTED IN THE VECTOR "PANORAMA".
NOTE: THE VECTOR PANORAMA2 IS USED IN ORDER TO AVOID THE SHADOWING PHENOMENA CAUSED BY THE SECOND CALL (IF PRESENT)
TO THE FUNCTION.
AT THE END OF THE FUNCTION, THE FINAL PANORAMA IMAGE (STORED INSIDE THE LAST POSITION OF PANORAMA) IS SHOWN
*/
void matching(std::vector<std::vector<cv::Mat>> all_channels, cv::Mat mask1, cv::Mat mask2, const std::string& descr_name) {

	cv::Mat img;
	std::vector<cv::Mat> panorama;
	for(int i = 0; i < 3; ++i) {
		cv::merge(all_channels[0], img);
 		cv::cvtColor(img, img, cv::COLOR_Lab2BGR);
	}
	panorama.push_back(img);

	cv::Mat descriptor1;				
	cv::Mat descriptor2;				
	std::vector<cv::KeyPoint> kp1{};	
	std::vector<cv::KeyPoint> kp2{};	
	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create(300, 3, 0.08, 10, 1.6);
	cv::Ptr<cv::ORB> orbPtr = cv::ORB::create(300, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

	std::vector<cv::DMatch> matches{};
	cv::Ptr<cv::BFMatcher> bf_matcher_euclidean = cv::BFMatcher::create(cv::NORM_L2, true);
	cv::Ptr<cv::BFMatcher> bf_matcher_hamming = cv::BFMatcher::create(cv::NORM_HAMMING, true);

	std::vector<std::vector<float>> set_translations(all_channels.size()-1, std::vector<float>(2));

	for(int i = 0; i < all_channels.size()-1; i++) {
		cv::Mat image1 = all_channels[i][0];
		cv::Mat image2 = all_channels[i+1][0];
		if(descr_name == "sift") {
			siftPtr->detectAndCompute(image1, mask1, kp1, descriptor1, false);
			siftPtr->detectAndCompute(image2, mask2, kp2, descriptor2, false);
			bf_matcher_euclidean->match(descriptor1, descriptor2, matches);
		}
		if(descr_name == "orb") {
			orbPtr->detectAndCompute(image1, mask1, kp1, descriptor1, false);
			orbPtr->detectAndCompute(image2, mask2, kp2, descriptor2, false);
			bf_matcher_hamming->match(descriptor1, descriptor2, matches);
		}
		int min = 2000;
		for(int i = 0; i < matches.size(); ++i) {
			if(matches[i].distance < min)
				min = matches[i].distance;	
		}
		for(int i = 0; i < matches.size(); ++i) {
			if(matches[i].distance > 2.5*min) {
				matches.erase(matches.begin() + i);
			}
		}	
		std::vector<float> single_delta = ransac(matches, kp1, kp2, image1);
		for(int j = 0; j < 2; ++j) {
			set_translations[i][j] = single_delta[j];
		}
	}

	int check = wrongOrder(set_translations, all_channels);
	std::vector<cv::Mat> panorama2;

	if(check) {
		matching(all_channels, mask1, mask2, descr_name);	
	}
	else {
		for(int i = 0; i < all_channels.size()-1; i++) {
			float warp[] = {1.0, 0.0, -set_translations[i][0], 0.0, 1.0, -set_translations[i][1]};
			cv::Mat translation_matrix = cv::Mat(2, 3, CV_32F, warp);
			std::vector<cv::Mat> temp;
			for(int j = 0; j < 3; ++j) {
				cv::Mat translated;
				warpAffine(all_channels[i+1][j], translated, translation_matrix, all_channels[i+1][j].size());
				temp.push_back(translated);
			}
			cv::Mat merged;
			cv::merge(temp, merged);		
  			cv::cvtColor(merged, merged, cv::COLOR_Lab2BGR);
			temp.clear(); 
			cv::Mat risultato2;
			merged(cv::Range(0, merged.rows), cv::Range(0, merged.cols-set_translations[i][0])).copyTo(risultato2);
			cv::Mat add;
			cv::hconcat(panorama[i], risultato2, add);
			panorama.push_back(add);
		}
		panorama2 = panorama;
		cv::Mat final_image = panorama2[panorama2.size()-1];
		cv::namedWindow("final image", cv::WINDOW_NORMAL);
		cv::imshow("final image", final_image);
		cv::waitKey(0);
	}
}

int main() {

	int num = 23;
	int angle = 33;
	std::vector<std::vector<cv::Mat>> total_set = loadAndProject(num, angle);
	std::vector<cv::Mat> masks = maskCreation(total_set[0][0]);

	matching(total_set, masks[0], masks[1], "sift");
	
}

/*
COMMENTED FUNCTION DEVELOPED FOR ANOTHER SCRIPT (SEE REPORT FOR MORE DETAILS).
BASICALLY IT DOES THE SAME THING THAT THE RANSAC FUNCTION DOES, BUT IT USES THE FINDHOMOGRAPHY FUNCTION.
TO GET THE SET OF INLIERS, A COMPARISON BETWEEN THE MASK RETURNED BY THE FUNCTION AND THE VECTOR OF MATCHES
IS DONE. IN PARTICULAR, IF THE Ith ROW OF THE MASK IS SET TO ONE, THEN THE Ith MATCH IS AN INLIER (I.E. A GOOD MATCH).
GIVEN THE SET OF INLIERS, THE FINAL TRANSLATION IS COMPUTED BY TAKING THE AVERAGE VALUE.
THE FUNCTION DOESN'T WORK FOR THIS SCRIPT, BUT I THOUGHT IT WOULD HAVE BEEN INTERESTING TO SEE THE DIFFERENCE 
BETWEEN THIS APPROACH AND THE MANUALLY IMPLEMENTED RANSAC. THE OUTPUT IS INSIDE THE FOLDER CALLED "RESULTS".
*/
/*void findhom(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2, cv::Mat image) {
	for(int i = 0; i < matches.size(); ++i) {
		obj.push_back(kp1[matches[i].queryIdx].pt);
		scene.push_back(kp2[matches[i].trainIdx].pt);
	}
	cv::Mat mask_hom;
	cv::Mat hom = cv::findHomography(obj, scene, cv::RANSAC, 3, mask_hom);
	std::vector<cv::DMatch> good_matches;
	for(int i = 0; i < matches.size(); ++i) {
		if((int)mask_hom.at<uchar>(i, 0) == 1) {
			good_matches.push_back(matches[i]);
		}
	}
	float x_tot = 0;
	float y_tot = 0;
	for(int j = 0; j < good_matches.size(); ++j) {
		float x1 = kp1[good_matches[j].queryIdx].pt.x;
	 	float x2 = kp2[good_matches[j].trainIdx].pt.x;
	 	float y1 = kp1[good_matches[j].queryIdx].pt.y;
	 	float y2 = kp2[good_matches[j].trainIdx].pt.y;
	 	float x_trans = x2 + (image.size().width - x1);
	 	float y_trans = y2 - y1;
	 	x_tot = x_tot + x_trans;
	 	y_tot = y_tot + y_trans;
	}
	float final_x = x_tot/good_matches.size();
	float final_y = y_tot/good_matches.size();
}*/

/*
COMMENTED FUNCTION DEVELOPED FOR ANOTHER SCRIPT (SEE REPORT FOR MORE DETAILS).
THIS FUNCTION PERFORMS THE STITCHING OPERATION STARTING FROM THE MATRIX OBTAINED THROUGH THE 
FINDHOMOGRAPHY FUNCTION. IT USES THE WARPPERSPECTIVE FUNCTION, THAT FINDS MORE COMPLEX TRANSFORMATION
(NOT ONLY SIMPLE TRANSLATIONS). THE RESULT IS VERY GOOD, BUT THE FUNCTION IS MUCH MORE DIFFICULT TO USE,
ESPECIALLY WHEN THE NUMBER OF IMAGES TO STITCH IS GREATER THAN TWO. FOR THIS REASON, IT WAS ABANDONED SOON.
THE FUNCTION DOESN'T WORK FOR THIS SCRIPT, BUT I THOUGHT IT WOULD HAVE BEEN INTERESTING TO LOOK AT THE VERY
GOOD RESULTS IT PROVIDES. THE OUTPUT IS INSIDE THE FOLDER CALLED "RESULTS".
*/
/*void perspective() {
	cv::Mat mask_hom;
	cv::Mat hom = cv::findHomography(scene, obj, cv::RANSAC, 3, mask_hom);
	cv::Mat result;
	cv::warpPerspective(image2, result, hom, cv::Size(image1.cols+image2.cols, image1.rows+10), cv::INTER_CUBIC);
	//RESULT CONTAINS THE IMAGE 2 THAT WAS TRANSFORMED IN ORDER TO BE MATCHED WITH IMAGE 1 
	cv::waitKey(0);
	cv::Mat final(cv::Size(image2.cols*2, image2.rows+10), CV_8UC1);
	cv::Mat roi3(final, cv::Rect(0, 0, image1.cols, image1.rows));
	cv::Mat roi4(final, cv::Rect(0, 0, result.cols, result.rows));
	result.copyTo(roi4);
	image1.copyTo(roi3);
}*/