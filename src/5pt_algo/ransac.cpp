#include "ransac.hpp"
#include "homo2fund.hpp"
#include "sift2homo.hpp"

using namespace Eigen;

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

bool ransac(int iterations, double threshold, double confidence,
            const vector<DMatch>& matches, const vector<KeyPoint>& keypoints_1,
            const vector<KeyPoint>& keypoints_2, int img_width, int img_height,
            Mat& F) {
  // create uniform distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  default_random_engine generator(seed);
  uniform_int_distribution<> rand_dist(0, matches.size() - 1);

  // start iterating
  long max_iterations = LONG_MAX;
  for (int iter = 0; iter < iterations; iter++) {
    // grab the points and angles in Mat
    Mat points_1(5, 2, CV_64F);
    Mat points_2(5, 2, CV_64F);
    Mat angles(5, 1, CV_64F);
    for (int i = 0; i < 5; i++) {
      int index = rand_dist(generator);
      points_1.at<double>(i, 0) = keypoints_1[matches[index].queryIdx].pt.x;
      points_1.at<double>(i, 1) = keypoints_1[matches[index].queryIdx].pt.y;
      points_2.at<double>(i, 0) = keypoints_2[matches[index].trainIdx].pt.x;
      points_2.at<double>(i, 1) = keypoints_2[matches[index].trainIdx].pt.y;
      angles.at<double>(i, 0) = keypoints_2[matches[index].trainIdx].angle -
                                keypoints_1[matches[index].queryIdx].angle;
    }
    // cout << points_1 << endl << points_2 << endl;

    // normalize
    Mat T_1, T_2;
    normalize(points_1, points_2, T_1, T_2);
    // if (normalize(points_1, points_2, T_1, T_2)) cout << "Normalized." <<
    // endl;

    // get homography
    Mat homography;
    Rect roi(0, 0, points_1.cols, 3);
    if (sift_to_homography(Mat(points_1, roi), Mat(points_2, roi), angles,
                           homography)) {
      homography = T_2.inv() * homography * T_1;
      // cout << "Found homography." << endl << homography << endl << endl;
    }

    // check the other correspondences
    bool points_too_close = false;
    for (int i = 3; i <= 4; i++) {
      Mat pt_1(3, 1, CV_64F);
      pt_1.at<double>(0, 0) = points_1.at<double>(i, 0);
      pt_1.at<double>(1, 0) = points_1.at<double>(i, 1);
      pt_1.at<double>(2, 0) = 1;
      Mat pt_2(3, 1, CV_64F);
      pt_2.at<double>(0, 0) = points_2.at<double>(i, 0);
      pt_2.at<double>(1, 0) = points_2.at<double>(i, 1);
      pt_2.at<double>(2, 0) = 1;

      Mat proj_1 = homography * pt_1;
      proj_1 = proj_1 / proj_1.at<double>(2, 0);
      if (norm(proj_1 - pt_2) < threshold) {
        points_too_close = true;
        break;
      }
    }
    if (points_too_close) continue;

    // get fundamental matrix
    Mat fundamental_matrix;
    if (!homography_to_fundamental(homography, points_1, points_2, img_width,
                                   img_height, fundamental_matrix))
      cout << "Fundamental matrix estimation failed." << endl;

    // collect inliers, by estimating symmetric epipolar distance
    // for each correspondences, estimate epipolar distance
    // update inliers, max iterations

    // cut off when max iterations achieved
    if (iter > max_iterations) break;
  }

  cout << "All done." << endl;
}
