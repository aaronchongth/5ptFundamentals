#include "ransac.hpp"
#include "sift2homo.hpp"
#include "data_handler/data_handler.hpp"
#include "utilities/utilities.hpp"

using namespace Eigen;

bool ransac(int iterations, double threshold, double confidence,
            const vector<DMatch>& matches, const vector<KeyPoint>& keypoints_1,
            const vector<KeyPoint>& keypoints_2, Mat& F) {
  // create uniform distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  uniform_int_distribution<> rand_dist(0, matches.size());

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
  if (normalize(points_1, points_2, T_1, T_2)) cout << "Normalized." << endl;

  // get homography
  Mat homography;
  Rect roi(0, 0, points_1.cols, 3);
  if (sift_to_homography(Mat(points_1, roi), Mat(points_2, roi), angles,
                         homography))
    cout << "Found homography." << endl;

  // check the other correspondences
  // get fundamental matrix
  // collect inliers, by estimating symmetric epipolar distance for
  // each correspondences

  cout << "All done." << endl;
}
