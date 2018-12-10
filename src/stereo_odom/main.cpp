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

  // initialize pose
  Mat pose(4, 1, CV_64F);
  for (int dof = 0; dof < 3; dof++) pose.at<double>(dof, 0) = 0;
  pose.at<double>(3, 0) = 1;

  // initialize the file to be written in
  ofstream myfile;
  myfile.open("7pt.txt");
  myfile << "0 0 0" << endl;

  // start looping, currently hard coding to loop through all images
  int n_frames = 83;
  int total_ms = 0;
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
    Mat F;
    // total_ms += get_5pt_F(img_1, keypoints_1, descriptors_1, img_2,
    // keypoints_2,
    //                       descriptors_2, good_matches, F);
    total_iterations += get_7pt_F(img_1, keypoints_1, descriptors_1, img_2,
                                  keypoints_2, descriptors_2, good_matches, F);

    // getting R and t
    double K_data[9] = {984.2439, 0, 690, 0, 980.8141, 233.1966, 0, 0, 1};
    Mat K(3, 3, CV_64F, K_data);
    Mat R(3, 3, CV_64F);
    Mat t(3, 1, CV_64F);
    get_Rt_from_F(keypoints_1, keypoints_2, good_matches, K, F, R, t);

    // checking
    // std::cout << R_5 << std::endl;
    // std::cout << t_5 << std::endl;

    // getting motion transformation matrix
    Mat T = Mat::eye(4, 4, CV_64F);
    Mat T_R = T(Rect(0, 0, R.cols, R.rows));
    R.copyTo(T_R);
    T.at<double>(0, 3) = t.at<double>(0, 0);
    T.at<double>(1, 3) = t.at<double>(1, 0);
    T.at<double>(2, 3) = t.at<double>(2, 0);

    Mat new_pose = T * pose;
    std::cout << new_pose << std::endl;

    // update new pose
    pose = new_pose.clone();

    // visualizing
    // namedWindow("Display window", WINDOW_AUTOSIZE);
    // imshow("Display window", img_2);
    // waitKey(1);

    // update to next frame
    img_1 = img_2.clone();

    // writing to text file
    // myfile << "Writing this to a file.\n";
    myfile << pose.at<double>(0, 0) << " " << pose.at<double>(1, 0) << " "
           << pose.at<double>(2, 0) << endl;
  }
  myfile.close();

  std::cout << "All done." << std::endl;
  std::cout << total_ms / n_frames << std::endl;
  return 0;
}