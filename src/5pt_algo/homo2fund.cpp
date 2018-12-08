#include "homo2fund.hpp"
#include "utilities/utilities.hpp"

using namespace std;

bool homography_to_fundamental(const Mat& H, const Mat& points_1,
                               const Mat& points_2, int img_width,
                               int img_height, Mat& F) {
  // hallucinate 5 correspondences on the image
  vector<double> hallucinated_pts = {0,
                                     0,
                                     1,
                                     (double)img_width,
                                     0,
                                     1,
                                     0,
                                     (double)img_height,
                                     1,
                                     (double)img_width,
                                     (double)img_height,
                                     1,
                                     (double)img_width / 2,
                                     (double)img_height / 2,
                                     1};
  Mat hallu_pts_1(hallucinated_pts);
  hallu_pts_1 = hallu_pts_1.reshape(1, 5);
  hallu_pts_1 = hallu_pts_1.t();
  Mat hallu_pts_2 = H * hallu_pts_1;

  // normalize everything in hallu pts 2
  for (int i = 0; i < 5; i++)
    hallu_pts_2.col(i) /= hallu_pts_2.at<double>(2, i);

  // put the points together, they should all be in image pixels
  // TODO: check normalization by basel
  Mat all_points_1, all_points_2;
  hconcat(hallu_pts_1, points_1.t(), all_points_1);
  hconcat(hallu_pts_2, points_2.t(), all_points_2);

  // solve using DLT, Af = 0
  return overconstrained_DLT(all_points_1, all_points_2, F);
}

bool check_F_signs(const Mat& F, const Mat& points_1, const Mat& points_2) {
  Mat W(10, 1, CV_64F);
  Mat U(10, 9, CV_64F);
  Mat Vt(9, 10, CV_64F);
  SVDecomp(F.t(), W, U, Vt, SVD::MODIFY_A + SVD::FULL_UV);

  Mat e1(3, 1, CV_64F);
  e1.push_back(Vt.col(2));

  cout << points_1 << endl << points_2 << endl;
  return true;
}