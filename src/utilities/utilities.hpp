#pragma once

#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

bool normalize(Mat& points_1, Mat& points_2, Mat& T_1, Mat& T_2);

unsigned int num_inliers(const std::vector<KeyPoint>& keypoints_1,
                         const std::vector<KeyPoint>& keypoints_2, const Mat& F,
                         const std::vector<DMatch>& good_matches,
                         const double threshold);

// not given anything, get all the stuff
bool get_matched_images(Mat& img1, std::vector<KeyPoint>& keypoints_1,
                        Mat& descriptors_1, Mat& img2,
                        std::vector<KeyPoint>& keypoints_2, Mat& descriptors_2,
                        std::vector<DMatch>& good_matches);

// given images, get all the other stuff
bool match_images(const Mat& img_1, const Mat& img_2,
                  std::vector<KeyPoint>& keypoints_1,
                  std::vector<KeyPoint>& keypoints_2, Mat& descriptors_1,
                  Mat& descriptors_2, std::vector<DMatch>& good_matches);

// points_1 and points_2 are in the form of 3 x n_points
// make sure n_points > 9
bool overconstrained_DLT(const Mat& points_1, const Mat& points_2, Mat& F);

bool plot_testing(const Mat& img_1, const Mat& img_2,
                  const std::vector<KeyPoint>& keypoints_1,
                  const std::vector<KeyPoint>& keypoints_2,
                  std::vector<DMatch>& good_matches, const Mat& F,
                  float threshold);

std::vector<Mat> run7Point(Mat _m1, Mat _m2);