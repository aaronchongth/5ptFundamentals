#include "ransac.hpp"
#include "utilities/utilities.hpp"
#include "utils.hpp"

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
  get_matched_images(img_1, keypoints_1, descriptors_1, img_2, keypoints_2,
                     descriptors_2, good_matches);

  auto t0 = chrono::system_clock::now();
  // parameters for GCRANSAC
  int iterations = 1000;
  float threshold = 0.1;
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

  // for plotting
  plot_testing(img_1, img_2, keypoints_1, keypoints_2, good_matches,
               fundamental_matrix, threshold);
}