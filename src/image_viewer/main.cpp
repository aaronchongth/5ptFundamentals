#include "cxxopts/cxxopts.hpp"
#include "data_handler/data_handler.hpp"

using namespace std;
using namespace cv;
namespace opt = cxxopts;

int main() {
  dataConfig data_config;
  KITTIDataHandler data_handler(data_config);
  Mat test_mat;

  for (int i = 0; i < 1000; i++) {
    bool status = data_handler.get_next_image(test_mat);

    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", test_mat);

    // the average frame rate from this KITTI dataset
    waitKey(100);
  }

  cout << "All done." << endl;
}