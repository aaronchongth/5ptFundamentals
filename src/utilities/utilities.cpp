#include "utilities.hpp"
#include <iostream>
#include "data_handler/data_handler.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void normColumn(Mat& m, unsigned int col, double& mn, double& avg_dist) {
  unsigned int n = m.rows;
  mn = 0;

  for (int i = 0; i < n; i++) {
    mn = mn + m.at<double>(i, col);
  }

  mn /= n;

  for (int i = 0; i < n; i++) {
    m.at<double>(i, col) = m.at<double>(i, col) - mn;
  }

  avg_dist = 0;

  for (int i = 0; i < n; i++) {
    avg_dist += abs(m.at<double>(i, col) - mn);
  }

  avg_dist /= n;

  for (int i = 0; i < n; i++) {
    m.at<double>(i, col) /= avg_dist;
  }
}

bool normalize(Mat& m1, Mat& m2, Mat& T1, Mat& T2) {
  double mn_x1, mn_y1, mn_x2, mn_y2;
  double avg_dist_x1, avg_dist_y1, avg_dist_x2, avg_dist_y2;

  normColumn(m1, 0, mn_x1, avg_dist_x1);
  normColumn(m1, 1, mn_y1, avg_dist_y1);
  normColumn(m2, 0, mn_x2, avg_dist_x2);
  normColumn(m2, 1, mn_y2, avg_dist_y2);

  double data11[9] = {1 / avg_dist_x1, 0, 0, 0, 1 / avg_dist_y1, 0, 0, 0, 1.0f};
  Mat scale1(3, 3, CV_64F, data11);
  double data12[9] = {1, 0, -mn_x1, 0, 1, -mn_y1, 0, 0, 1};
  Mat shift1(3, 3, CV_64F, data12);

  double data21[9] = {1 / avg_dist_x2, 0, 0, 0, 1 / avg_dist_y2, 0, 0, 0, 1.0f};
  Mat scale2(3, 3, CV_64F, data21);
  double data22[9] = {1, 0, -mn_x2, 0, 1, -mn_y2, 0, 0, 1};
  Mat shift2(3, 3, CV_64F, data22);

  T1 = scale1 * shift1;
  T2 = scale2 * shift2;

  // return the homogeneous coordinates of normalized points
  unsigned int n = m1.rows;
  hconcat(m1, Mat(n, 1, CV_64F, Scalar(1)), m1);
  hconcat(m2, Mat(n, 1, CV_64F, Scalar(1)), m2);

  return true;
}

bool get_matched_images(Mat& img_1, std::vector<KeyPoint>& keypoints_1,
                        Mat& descriptors_1, Mat& img_2,
                        std::vector<KeyPoint>& keypoints_2, Mat& descriptors_2,
                        std::vector<DMatch>& good_matches) {
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

// Does DLT, not to be confused with BLT which is also nice
bool overconstrained_DLT(const Mat& points_1, const Mat& points_2, Mat& F) {
  // solve for Af = 0
  // construct the main A matrix
  Mat A(points_1.cols, 9, CV_64F);
  for (int i = 0; i < points_1.cols; i++) {
    double u1 = points_1.at<double>(0, i);
    double v1 = points_1.at<double>(1, i);
    double u2 = points_2.at<double>(0, i);
    double v2 = points_2.at<double>(1, i);

    A.at<double>(i, 0) = u1 * u2;
    A.at<double>(i, 1) = v1 * u2;
    A.at<double>(i, 2) = u2;
    A.at<double>(i, 3) = u1 * v2;
    A.at<double>(i, 4) = v1 * v2;
    A.at<double>(i, 5) = v2;
    A.at<double>(i, 6) = u1;
    A.at<double>(i, 7) = v1;
    A.at<double>(i, 8) = 1;
  }

  // this function assumes A is overconstrained, so only 1 solution to F
  Mat W(points_1.cols, 1, CV_64F);
  Mat U(points_1.cols, 9, CV_64F);
  Mat Vt(9, points_1.cols, CV_64F);
  Mat f(9, 1, CV_64F);
  SVDecomp(A, W, U, Vt, SVD::MODIFY_A + SVD::FULL_UV);
  f.push_back(Vt.col(5));
  F = f.reshape(1, 3);

  return true;
}