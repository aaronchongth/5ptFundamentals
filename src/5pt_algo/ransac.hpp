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

bool ransac(int iterations, double threshold, double confidence,
            const vector<DMatch>& matches, const vector<KeyPoint>& keypoints_1,
            const vector<KeyPoint>& keypoints_2, int img_width, int img_height,
            Mat& F);