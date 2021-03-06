#include "5pt.hpp"
#include "ransac.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace cv::xfeatures2d;
namespace opt = cxxopts;

int get_5pt_F(Mat& img_1, std::vector<KeyPoint>& keypoints_1,
              Mat& descriptors_1, Mat& img_2,
              std::vector<KeyPoint>& keypoints_2, Mat& descriptors_2,
              std::vector<DMatch>& good_matches, Mat& fund) {
  auto t0 = chrono::system_clock::now();
  // parameters for GCRANSAC
  int iterations = 1000;
  float threshold = 0.1;
  float confidence = 0.95;
  Mat fundamental_matrix(3, 3, CV_64F);
  // start ransac loop
  if (!ransac(iterations, threshold, confidence, good_matches, keypoints_1,
              keypoints_2, img_2.cols, img_2.rows, fundamental_matrix))
    cout << "Ransac failed." << endl;

  // end timing
  auto duration = chrono::duration_cast<chrono::milliseconds>(
      chrono::system_clock::now() - t0);
  int ms_passed = (int)duration.count();
  cout << "Found F matrix in " << ms_passed << " milliseconds" << endl;

  // returning the fundamental matrix
  fund = fundamental_matrix;

  return ms_passed;
}