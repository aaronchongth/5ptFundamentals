#include "utils.hpp"

using namespace cv;

void get_Rt_from_F(const std::vector<KeyPoint>& keypoints_1,
                   const std::vector<KeyPoint>& keypoints_2,
                   const std::vector<DMatch>& good_matches, const Mat& K,
                   const Mat& F, Mat& R, Mat& t) {
  Mat E(3, 3, CV_64F);
  sfm::essentialFromFundamental(F, K, K, E);
  std::vector<Mat> Rs, ts;

  const DMatch* this_match = &good_matches[5];
  Point2f pt1 = keypoints_1[this_match->queryIdx].pt;
  Point2f pt2 = keypoints_2[this_match->trainIdx].pt;

  double pt1_d[2] = {pt1.x, pt1.y};
  double pt2_d[2] = {pt2.x, pt2.y};
  Mat pt1_m(2, 1, CV_64F, pt1_d);
  Mat pt2_m(2, 1, CV_64F, pt2_d);

  sfm::motionFromEssential(E, Rs, ts);
  int sol = sfm::motionFromEssentialChooseSolution(Rs, ts, K, pt1_m, K, pt2_m);

  R = Rs[sol];
  t = ts[sol];
}

int main() {
  static dataConfig data_config;
  static KITTIDataHandler data_handler(data_config);
  Mat img_1, img_2;

  // get the starting frame
  if (!data_handler.get_next_image(img_1)) cry("Getting first image failed.");

  // start looping, currently hard coding to loop through all images
  int n_frames = 83;
  for (int i = 0; i < n_frames; i++) {
    // get the next frame
    if (!data_handler.get_next_image(img_2)) cry("Getting next image failed.");

    // do matching
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    std::vector<DMatch> good_matches;
    if (!match_images(img_1, img_2, keypoints_1, keypoints_2, descriptors_1,
                      descriptors_2, good_matches))
      cry("Getting matches failed.");

    // do the algorithm
    Mat F_5;
    get_5pt_F(img_1, keypoints_1, descriptors_1, img_2, keypoints_2,
              descriptors_2, good_matches, F_5);

    // getting R and t
    double K_data[9] = {984.2439, 0, 690, 0, 980.8141, 233.1966, 0, 0, 1};
    Mat K(3, 3, CV_64F, K_data);
    Mat R_5(3, 3, CV_64F);
    Mat t_5(3, 1, CV_64F);
    get_Rt_from_F(keypoints_1, keypoints_2, good_matches, K, F_5, R_5, t_5);

    // checking
    std::cout << R_5 << std::endl;
    std::cout << t_5 << std::endl << std::endl;

    // update to next frame
    img_1 = img_2.clone();
  }

  std::cout << "All done." << std::endl;
  return 0;
}