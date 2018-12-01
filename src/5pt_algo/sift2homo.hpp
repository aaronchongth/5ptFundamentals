#pragma once

// STL
#include <fstream>
#include <iostream>
#include <vector>

// opencv
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

bool sift_to_homography(const vector<KeyPoint>& img_1_pts,
                        const vector<KeyPoint>& img_2_pts,
                        const vector<double>& angles, MatrixXd& homography);