#include "sift2homo.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace cv::xfeatures2d;
namespace opt = cxxopts;

int main() {
  dataConfig data_config;
  KITTIDataHandler data_handler(data_config);

  // get first 2 images
  Mat img_1;
  Mat img_2;

  if (data_handler.get_next_image(img_1)) cout << endl;
  if (data_handler.get_next_image(img_2)) cout << endl;

  // find SIFT
  int minHessian = 1000;
  Ptr<SURF> detector = SURF::create(minHessian);
  vector<KeyPoint> keypoints_1, keypoints_2;
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // test
  cout << keypoints_1.size() << " " << keypoints_2.size() << endl;
  cout << keypoints_1[0].angle << endl;
  cout << keypoints_1[1].angle << endl;
  cout << keypoints_1[2].angle << endl;

  // draw keypoints
  Mat img_keypoints_1;
  Mat img_keypoints_2;
  drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
                DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1),
                DrawMatchesFlags::DEFAULT);

  // show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1);
  imshow("Keypoints 2", img_keypoints_2);
  waitKey(0);

  // parameters for GCRANSAC
  int iterations = 2000;
  float threshold = 3.0;
  float confidence = 0.99;
  vector<pair<KeyPoint, KeyPoint>> inliers;
  Matrix3f fundamental_matrix;

  if (ransac(iterations, threshold, confidence, keypoints_1, keypoints_2,
             inliers, fundamental_matrix)) {
    cout << "ransac done." << endl;
  }

  // create uniform distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  uniform_int_distribution<> rand_dist_1(0, keypoints_1.size());
  uniform_int_distribution<> rand_dist_2(0, keypoints_2.size());
  for (int i = 0; i < 10; i++) {
    cout << rand_dist_1(generator) << " " << rand_dist_2(generator) << endl;
  }

  cout << "All done." << endl;
}