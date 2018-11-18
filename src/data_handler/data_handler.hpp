#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

inline string locateFile(const string &input,
                         const vector<string> &directories) {
  string file;
  const int MAX_DEPTH{10};
  bool found{false};
  for (auto &dir : directories) {
    file = dir + input;
    for (int i = 0; i < MAX_DEPTH && !found; i++) {
      ifstream checkFile(file);
      found = checkFile.is_open();
      if (found) break;
      file = "../" + file;
    }
    if (found) break;
    file.clear();
  }
  return file;
}

struct dataConfig {
  string data_path = "data/2011_09_26/2011_09_26_drive_0002_extract/image_00/";
  string data_ext = "png";
};

// use this to go through 1 specific data folder, change the path for others
class KITTIDataHandler {
 public:
  explicit KITTIDataHandler(const dataConfig &data_config);
  ~KITTIDataHandler();

  // high level member functions
  bool get_next_image(Mat &image);

  // util member functions
  string get_data_path() { return data_path_; }

 private:
  string data_path_, data_ext_;
  vector<string> directories_;
  int counter_;
};