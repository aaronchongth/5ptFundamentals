// #include "ransac.hpp"
// #include "sift2homo.hpp"
#include "utils.hpp"
#include "utilities/utilities.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace cv::xfeatures2d;
namespace opt = cxxopts;

int main() {
  Mat img_1, img_2;
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  std::vector<DMatch> good_matches;
  get_matched_images(img_1, keypoints_1, descriptors_1, 
                     img_2, keypoints_2, descriptors_2,
                     good_matches);

  auto t0 = chrono::system_clock::now();
  // parameters for GCRANSAC
  int iterations = 1000;
  float threshold = 3.0;
  float confidence = 0.99;
  Mat fundamental_matrix(3, 3, CV_64F);
  // start ransac loop
  if (ransac(iterations, threshold, confidence, good_matches, keypoints_1,
             keypoints_2, fundamental_matrix)) {
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
  // drawKeypoints(img_1, keypoints_1, img_keypoints_1,
  // Scalar::all(-1),
  //               DrawMatchesFlags::DEFAULT);
  // drawKeypoints(img_2, keypoints_2, img_keypoints_2,
  // Scalar::all(-1),
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