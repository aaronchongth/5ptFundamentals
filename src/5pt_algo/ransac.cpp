#include "ransac.hpp"
#include "data_handler/data_handler.hpp"
#include "homo2fund.hpp"
#include "sift2homo.hpp"
#include "utilities/utilities.hpp"

using namespace Eigen;

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

    // normalize
    Mat T_1, T_2;
    normalize(points_1, points_2, T_1, T_2);

    // get homography
    Mat homography;
    Rect roi(0, 0, points_1.cols, 3);
    if (sift_to_homography(Mat(points_1, roi), Mat(points_2, roi), angles,
                           homography)) {
      homography = T_2.inv() * homography * T_1;
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

    std::cout << "got here" << std::endl;
    std::cout << fundamental_matrix << std::endl;
    std::cout << "rows: "<< fundamental_matrix.rows << " and cols: " << fundamental_matrix.cols << std::endl;
    unsigned int n_inliers;
    double inlier_threshold = 0.01;
    n_inliers = num_inliers(keypoints_1, keypoints_2, fundamental_matrix, matches, inlier_threshold);
    std::cout << "inliers: " << n_inliers << std::endl;

    // collect inliers, by estimating symmetric epipolar distance
    // for each correspondences, estimate epipolar distance
    // update inliers, max iterations

    // cut off when max iterations achieved
    if (iter > max_iterations) break;
  }
}
