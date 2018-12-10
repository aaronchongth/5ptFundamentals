#include "homo2fund.hpp"
#include "utilities/utilities.hpp"

using namespace std;

std::vector<Mat> homography_to_fundamental(const Mat& H, const Mat& points_1,
                                           const Mat& points_2, int img_width,
                                           int img_height, Mat& F) {
  // hallucinate 5 correspondences on the image
  // vector<double> hallucinated_pts = {0,
  //                                    0,
  //                                    1,
  //                                    (double)img_width,
  //                                    0,
  //                                    1,
  //                                    0,
  //                                    (double)img_height,
  //                                    1,
  //                                    (double)img_width,
  //                                    (double)img_height,
  //                                    1,
  //                                    (double)img_width / 2,
  //                                    (double)img_height / 2,
  //                                    1};
  vector<double> hallucinated_pts = {
      0, 0, 1, (double)img_width, (double)img_height, 1};
  Mat hallu_pts_1(hallucinated_pts);
  // hallu_pts_1 = hallu_pts_1.reshape(1, 5);
  hallu_pts_1 = hallu_pts_1.reshape(1, 2);
  hallu_pts_1 = hallu_pts_1.t();
  Mat hallu_pts_2 = H * hallu_pts_1;

  // normalize everything in hallu pts 2
  // for (int i = 0; i < 5; i++)
  for (int i = 0; i < 2; i++)
    hallu_pts_2.col(i) /= hallu_pts_2.at<double>(2, i);

  // put the points together, they should all be in image pixels
  // TODO: check normalization by basel
  Mat homo_points_1(points_1.rows, 3, CV_64F);
  Mat homo_points_2(points_2.rows, 3, CV_64F);
  for (int i = 0; i < points_1.rows; i++) {
    homo_points_1.at<double>(i, 0) = points_1.at<double>(i, 0);
    homo_points_1.at<double>(i, 1) = points_1.at<double>(i, 1);
    homo_points_1.at<double>(i, 2) = 1;
    homo_points_2.at<double>(i, 0) = points_2.at<double>(i, 0);
    homo_points_2.at<double>(i, 1) = points_2.at<double>(i, 1);
    homo_points_2.at<double>(i, 2) = 1;
  }

  Mat all_points_1, all_points_2;
  hconcat(hallu_pts_1, homo_points_1.t(), all_points_1);
  hconcat(hallu_pts_2, homo_points_2.t(), all_points_2);
  Rect only_xy_crop = Rect(0, 0, 7, 2);

  Mat p_1 = all_points_1(only_xy_crop);
  Mat p_2 = all_points_2(only_xy_crop);
  p_1 = p_1.t();
  p_2 = p_2.t();

  // solve using DLT, Af = 0
  // return overconstrained_DLT(all_points_1, all_points_2, F);

  // solve using 7pt algorithm
  return run7Point(p_1.clone(), p_2.clone());
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