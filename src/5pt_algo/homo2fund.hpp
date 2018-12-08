#pragma once

// STL
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

// opencv
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

// Eigen
#include "Eigen/Dense"

using namespace std;
using namespace cv;

bool check_F_signs(const Mat& f, const Mat& x1, const Mat& x2);

bool homography_to_fundamental(const Mat& H, const Mat& points_1,
                               const Mat& points_2, int img_width,
                               int img_height, Mat& F);