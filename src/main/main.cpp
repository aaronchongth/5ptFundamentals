#include <stdio.h>
#include <iostream>
#include <vector>
#include "5pt_algo/5pt.hpp"
#include "7pt_algo/7pt.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/sfm.hpp"
#include "utils.hpp"
#include "utilities/utilities.hpp"
#include "utils.hpp"
using namespace cv;
using namespace cv::xfeatures2d;

void get_Rt_from_F(const std::vector<KeyPoint>& keypoints_1, const std::vector<KeyPoint>& keypoints_2,
              const std::vector<DMatch>& good_matches, const Mat& K, const Mat& F, Mat& R, Mat& t)
{
  Mat E(3,3,CV_64F);
  sfm::essentialFromFundamental(F, K, K, E);
  std::vector<Mat> Rs, ts;

  const DMatch* this_match = &good_matches[5];
  Point2f pt1 = keypoints_1[this_match->queryIdx].pt;
  Point2f pt2 = keypoints_2[this_match->trainIdx].pt;

  double pt1_d[2] = {pt1.x, pt1.y};
  double pt2_d[2] = {pt2.x, pt2.y};
  Mat pt1_m(2,1,CV_64F, pt1_d);
  Mat pt2_m(2,1,CV_64F, pt2_d);

  sfm::motionFromEssential(E, Rs, ts);
  int sol = sfm::motionFromEssentialChooseSolution(Rs, ts, K, pt1_m, K, pt2_m);

  R = Rs[sol];
  t = ts[sol];
}

/*
 * @function main
 * @brief Main function
 */
int main(int argc, char** argv) {
  Mat img_1, img_2;
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  std::vector<DMatch> good_matches;
  get_matched_images(img_1, keypoints_1, descriptors_1, img_2, keypoints_2,
                     descriptors_2, good_matches);

  Mat F_7, F_5;
  get_7pt_F(img_1, keypoints_1, descriptors_1, img_2, keypoints_2,
            descriptors_2, good_matches, F_7);

  get_5pt_F(img_1, keypoints_1, descriptors_1, img_2, keypoints_2,
            descriptors_2, good_matches, F_5);

  double K_data[9] = {984.2439, 0, 690, 0, 980.8141, 233.1966, 0, 0, 1};
  Mat K(3, 3, CV_64F, K_data);
  Mat R_7(3, 3, CV_64F), R_5(3, 3, CV_64F);
  Mat t_7(3, 1, CV_64F), t_5(3, 1, CV_64F);
  get_Rt_from_F(keypoints_1, keypoints_2, good_matches, K, F_7, R_7, t_7);
  get_Rt_from_F(keypoints_1, keypoints_2, good_matches, K, F_5, R_5, t_5);

  std::cout << R_7 << std::endl;
  std::cout << R_5 << std::endl;
  std::cout << t_7 << std::endl;
  std::cout << t_5 << std::endl;
}