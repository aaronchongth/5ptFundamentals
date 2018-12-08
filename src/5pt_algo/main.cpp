#include "ransac.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace cv::xfeatures2d;
namespace opt = cxxopts;

int main() {
  dataConfig data_config;
  KITTIDataHandler data_handler(data_config);

  // get first images
  Mat img_1;
  if (data_handler.get_next_image(img_1)) cout << endl;

  // get the first set of SIFT and descriptors
  int n_features = 1000;
  Ptr<SIFT> detector = SIFT::create(n_features);
  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);

  // start timing and handle second image
  auto t0 = chrono::system_clock::now();
  Mat img_2;
  if (data_handler.get_next_image(img_2)) cout << endl;
  cout << img_2.rows << " " << img_2.cols << endl;
  detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

  // // matching using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector<std::vector<DMatch> > matches;
  matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

  // get the good matches
  std::vector<DMatch> good_matches;
  float ratio = 0.5f;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i][0].distance < ratio * matches[i][1].distance) {
      good_matches.push_back(matches[i][0]);
    }
  }

  // parameters for GCRANSAC
  int iterations = 1000;
  float threshold = 3.0;
  float confidence = 0.99;
  Mat fundamental_matrix(3, 3, CV_64F);
  // start ransac loop
  if (ransac(iterations, threshold, confidence, good_matches, keypoints_1,
             keypoints_2, img_2.cols, img_2.rows, fundamental_matrix)) {
    cout << "ransac done." << endl;
  }

  // end timing
  auto duration = chrono::duration_cast<chrono::milliseconds>(
      chrono::system_clock::now() - t0);
  int ms_passed = (int)duration.count();
  cout << "One iteration takes: " << ms_passed << " milliseconds" << endl;
  // cout << "Number of matches found: " << matches.size() << endl;

  // visualize the matches
  // Mat img_matches;
  // drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches,
  //             Scalar::all(-1), Scalar::all(-1), vector<char>(),
  //             DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  // imshow("Good Matches", img_matches);
  // waitKey(0);

  // test
  // cout << keypoints_1.size() << " " << keypoints_2.size() <<
  // endl; cout << keypoints_1[0].angle << endl; cout <<
  // keypoints_1[1].angle << endl; cout << keypoints_1[2].angle <<
  // endl;

  // // draw keypoints
  // Mat img_keypoints_1;
  // Mat img_keypoints_2;
  // drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
  //               DrawMatchesFlags::DEFAULT);
  // drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1),
  //               DrawMatchesFlags::DEFAULT);

  // // show detected (drawn) keypoints
  // imshow("Keypoints 1", img_keypoints_1);
  // imshow("Keypoints 2", img_keypoints_2);
  // waitKey(0);

  // // parameters for GCRANSAC
  // int iterations = 2000;
  // float threshold = 3.0;
  // float confidence = 0.99;
  // vector<pair<KeyPoint, KeyPoint>> inliers;
  // Matrix3f fundamental_matrix;

  // if (ransac(iterations, threshold, confidence,
  // keypoints_1, keypoints_2,
  //            inliers, fundamental_matrix)) {
  //   cout << "ransac done." << endl;
  // }

  cout << "All done." << endl;
}