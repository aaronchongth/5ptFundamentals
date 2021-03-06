#include "data_handler.hpp"

using namespace std;
using namespace cv;

KITTIDataHandler::KITTIDataHandler(const dataConfig& data_config)
    : data_path_(data_config.data_path),
      data_ext_(data_config.data_ext),
      counter_(0) {
  directories_.push_back(data_path_ + "data/");
  directories_.push_back(data_path_);
  cout << "Data handler init done." << endl;
}

KITTIDataHandler::~KITTIDataHandler() {
  cout << "Data handler destructor called." << endl;
}

bool KITTIDataHandler::get_next_image(Mat& image) {
  // turn counter into 10 digits
  string image_name = to_string(counter_);
  while (image_name.length() < 10) image_name = '0' + image_name;
  image_name += '.' + data_ext_;

  // check if the image is available
  string image_path = locateFile(image_name, directories_);

  // printing stuff
  // cout << '\r' << "Collecting: " << image_name << flush;

  // read the image or loop around
  if (!image_path.empty()) {
    image = imread(image_path);
    counter_++;
    return true;
  } else {
    counter_ = 0;
    return true;
  }
}
