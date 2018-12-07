#include "utilities.hpp"

// using namespace Eigen;
using namespace cv;

bool normalize(Mat& points_1, Mat& points_2, Mat& T_1, Mat& T_2) {
  // Mat row_mean, col_mean;
  // reduce(img, row_mean, 0, CV_REDUCE_AVG);
  // reduce(img, col_mean, 1, CV_REDUCE_AVG);

  Mat points_1_mean, points_2_mean;
  reduce(points_1, points_1_mean, 0, CV_REDUCE_AVG);
  reduce(points_2, points_2_mean, 0, CV_REDUCE_AVG);
  for (int i = 0; i < points_1.rows; i++) {
    points_1.row(i) -= points_1_mean;
    points_2.row(i) -= points_2_mean;
  }

  double avg_dist_1 = 0;
  double avg_dist_2 = 0;
  for (int i = 0; i < 5; i++) {
    avg_dist_1 += norm(points_1.row(i));
    avg_dist_2 += norm(points_2.row(i));
  }
  avg_dist_1 /= 5;
  avg_dist_2 /= 5;
  double avg_ratio_1 = sqrt(2) / avg_dist_1;
  double avg_ratio_2 = sqrt(2) / avg_dist_2;

  points_1 *= avg_ratio_1;
  points_2 *= avg_ratio_2;

  double avg_arr_1[9] = {avg_ratio_1, 0, 0, 0, avg_dist_1, 0, 0, 0, 1};
  double mass_arr_1[9] = {1, 0, -points_1_mean.at<double>(0),
                          0, 1, -points_1_mean.at<double>(1),
                          0, 0, 1};
  double avg_arr_2[9] = {avg_ratio_2, 0, 0, 0, avg_dist_2, 0, 0, 0, 1};
  double mass_arr_2[9] = {1, 0, -points_2_mean.at<double>(0),
                          0, 1, -points_2_mean.at<double>(1),
                          0, 0, 1};
  Mat avg_mat_1(3, 3, CV_64F, avg_arr_1);
  Mat mass_mat_1(3, 3, CV_64F, mass_arr_1);
  Mat avg_mat_2(3, 3, CV_64F, avg_arr_2);
  Mat mass_mat_2(3, 3, CV_64F, mass_arr_2);

  // return the trasform matrices
  T_1 = avg_mat_1 * mass_mat_1;
  T_2 = avg_mat_2 * mass_mat_2;

  // return the homogeneous coordinates of normalized points
  hconcat(points_1, Mat(5, 1, CV_64F, Scalar(1)), points_1);
  hconcat(points_2, Mat(5, 1, CV_64F, Scalar(1)), points_2);

  return true;
}