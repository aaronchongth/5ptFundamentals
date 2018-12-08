#include <vector>
#include "opencv2/core.hpp"

using namespace cv;

int get_7pt_F(Mat& img1, std::vector<KeyPoint>& keypoints_1,
              Mat& descriptors_1, Mat& img2,
              std::vector<KeyPoint>& keypoints_2, Mat& descriptors_2,
              std::vector<DMatch>& good_matches, Mat& fund);