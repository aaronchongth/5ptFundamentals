#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "utils.hpp"
#include "utilities/utilities.hpp"
using namespace cv;
using namespace cv::xfeatures2d;

/*
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  Mat img_1, img_2;
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  std::vector<DMatch> good_matches;
  get_matched_images(img_1, keypoints_1, descriptors_1, 
                     img_2, keypoints_2, descriptors_2,
                     good_matches);
}