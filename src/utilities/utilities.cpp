#include "utilities.hpp"
#include <iostream>
#include "data_handler/data_handler.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "math.h"

using namespace cv;
using namespace cv::xfeatures2d;

void normColumn(Mat& m, unsigned int col, double& mn, double& avg_dist)
{
  unsigned int n = m.rows;
  mn = 0;

  for (int i = 0; i < n; i++)
  {
    mn = mn + m.at<double>(i,col);
  }

  mn /= n;


  for (int i = 0; i < n; i++)
  {
    m.at<double>(i,col) = m.at<double>(i,col) - mn;
  }

  avg_dist = 0;

  for (int i = 0; i < n; i++)
  {
    avg_dist += abs(m.at<double>(i,col) - mn);
  }

  avg_dist /= n;

  for (int i = 0; i < n; i++)
  {
    m.at<double>(i,col)  /= avg_dist;
  }
}

bool normalize(Mat& m1, Mat& m2, Mat& T1, Mat& T2)
{
  double mn_x1, mn_y1, mn_x2, mn_y2;
  double avg_dist_x1, avg_dist_y1, avg_dist_x2, avg_dist_y2;

  normColumn(m1, 0, mn_x1, avg_dist_x1);
  normColumn(m1, 1, mn_y1, avg_dist_y1);
  normColumn(m2, 0, mn_x2, avg_dist_x2);
  normColumn(m2, 1, mn_y2, avg_dist_y2);

  double data11[9] = {1/avg_dist_x1, 0, 0, 0, 1/avg_dist_y1, 0, 0, 0, 1.0f};
  Mat scale1( 3, 3, CV_64F, data11 );
  double data12[9] = {1, 0, -mn_x1, 0, 1, -mn_y1, 0, 0, 1};
  Mat shift1( 3, 3, CV_64F, data12 );

  double data21[9] = {1/avg_dist_x2, 0, 0, 0, 1/avg_dist_y2, 0, 0, 0, 1.0f};
  Mat scale2( 3, 3, CV_64F, data21 );
  double data22[9] = {1, 0, -mn_x2, 0, 1, -mn_y2, 0, 0, 1};
  Mat shift2( 3, 3, CV_64F, data22 );

  T1 = scale1 * shift1;
  T2 = scale2 * shift2;

  // return the homogeneous coordinates of normalized points
  unsigned int n = m1.rows;
  hconcat(m1, Mat(n, 1, CV_64F, Scalar(1)), m1);
  hconcat(m2, Mat(n, 1, CV_64F, Scalar(1)), m2);

  return true;
}

bool get_matched_images(Mat& img_1, std::vector<KeyPoint>& keypoints_1, Mat& descriptors_1, 
                        Mat& img_2, std::vector<KeyPoint>& keypoints_2, Mat& descriptors_2,
                        std::vector<DMatch>& good_matches)
{
  static dataConfig data_config;
  static KITTIDataHandler data_handler(data_config);

  // get first images
  if (data_handler.get_next_image(img_1)) std::cout << std::endl;

  // get the first set of SIFT and descriptors
  int n_features = 1000;
  Ptr<SIFT> detector = SIFT::create(n_features);
  detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);

  // start timing and handle second image
  if (data_handler.get_next_image(img_2)) std::cout << std::endl;
  detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

  // matching using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector<std::vector<DMatch> > matches;
  matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

  // get the good matches
  float ratio = 0.5f;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i][0].distance < ratio * matches[i][1].distance) {
      good_matches.push_back(matches[i][0]);
    }
  }

  return true;
}

unsigned int num_inliers(const std::vector<KeyPoint>& keypoints_1, const std::vector<KeyPoint>& keypoints_2, 
                         const Mat& F, const std::vector<DMatch>& good_matches, 
                         const double threshold)
{
  // check each point to see if it's an inlier
  int n_inliers = 0;
  int n_matches = good_matches.size();
  for (int j = 0; j < n_matches; j++)
  {
    const DMatch* this_match = &good_matches[j];
    Point2f pt1 = keypoints_1[this_match->queryIdx].pt;
    Point2f pt2 = keypoints_2[this_match->trainIdx].pt;

    double pt1_d[3] = {pt1.x, pt1.y, 1};
    Mat pt1_m(3,1,CV_64F, pt1_d);
    double pt2_d[3] = {pt2.x, pt2.y, 1};
    Mat pt2_m(3,1,CV_64F, pt2_d);

    // get epipolar lines and normalize
    Mat l1(3,1,CV_64F);
    Mat l2(3,1,CV_64F);
    l1 = pt2_m.t() * F;
    l1 /= std::sqrt(l1.at<double>(0,0)*l1.at<double>(0,0) +
                    l1.at<double>(0,1)*l1.at<double>(0,1) +
                    l1.at<double>(0,2)*l1.at<double>(0,2));
    l2 = F * pt1_m;
    l2 /= std::sqrt(l2.at<double>(0,0)*l2.at<double>(0,0) +
                    l2.at<double>(1,0)*l2.at<double>(1,0) +
                    l2.at<double>(2,0)*l2.at<double>(2,0));

    Mat e1 = l1 * pt1_m;
    Mat e2 = pt2_m.t() * l2;
    double dist = (abs(e1.at<double>(0,0)) + 
                    abs(e2.at<double>(0,0))) / 2.0f;

    if (dist < threshold) n_inliers++;
  } // end for each point

  return n_inliers;
}