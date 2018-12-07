#pragma once

// STL
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

// OpenCV
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

// Eigen
#include "Eigen/Dense"

using namespace std;
using namespace cv;
using namespace Eigen;

inline void cry(std::string error_msg) {
  std::ostringstream err;
  err << error_msg;
  throw std::runtime_error(err.str().c_str());
}

bool normalize(Mat& points_1, Mat& points_2, Mat& T_1, Mat& T_2);

bool ransac(int iterations, double threshold, double confidence,
            const vector<DMatch>& matches, const vector<KeyPoint>& keypoints_1,
            const vector<KeyPoint>& keypoints_2, Mat& F);