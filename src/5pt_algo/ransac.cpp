#include "ransac.hpp"

bool ransac(int iterations, double threshold, double confidence,
            const vector<KeyPoint>& keypoints_1,
            const vector<KeyPoint>& keypoints_2,
            vector<pair<KeyPoint, KeyPoint>>& inliers, Matrix3f& F) {
  // create uniform distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  uniform_int_distribution<> rand_dist_1(0, keypoints_1.size());
  uniform_int_distribution<> rand_dist_2(0, keypoints_2.size());

  // grab the points
  // normalize
  //

  cout << "All done." << endl;
}