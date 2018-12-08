#include "utils.hpp"

using namespace cv;

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

    // update to next frame
    img_1 = img_2.clone();
  }

  std::cout << "All done." << std::endl;
  return 0;
}