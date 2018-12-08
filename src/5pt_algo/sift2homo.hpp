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

bool sift_to_homography(const Mat& points_1, const Mat& points_2,
                        const Mat& angles, Mat& homography);

bool homography_to_affine(const Mat& H, double x1, double y1, double x2,
                          double y2, Mat& A);

bool decompose_affine(const Mat& A, double& sx, double& sy, double& alpha,
                      double& w);