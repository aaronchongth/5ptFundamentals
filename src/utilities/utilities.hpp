#pragma once

#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

bool normalize(Mat& points_1, Mat& points_2, Mat& T_1, Mat& T_2);

bool get_matched_images(Mat& img1, std::vector<KeyPoint>& keypoints_1,
                        Mat& descriptors_1, Mat& img2,
                        std::vector<KeyPoint>& keypoints_2, Mat& descriptors_2,
                        std::vector<DMatch>& good_matches);

// points_1 and points_2 are in the form of 3 x n_points
// make sure n_points > 9
bool overconstrained_DLT(const Mat& points_1, const Mat& points_2, Mat& F);
