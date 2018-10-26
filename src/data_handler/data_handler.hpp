#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"

using namespace std;

// inline std::string locateFile(const std::string &input,
//                               const std::vector<std::string> &directories) {
//   std::string file;
//   const int MAX_DEPTH{10};
//   bool found{false};
//   for (auto &dir : directories) {
//     file = dir + input;
//     for (int i = 0; i < MAX_DEPTH && !found; i++) {
//       std::ifstream checkFile(file);
//       found = checkFile.is_open();
//       if (found) break;
//       file = "../" + file;
//     }
//     if (found) break;
//     file.clear();
//   }

//   assert(!file.empty() &&
//          "Could not find a file due to it not existing in the data
//          directory.");
//   return file;
// }

struct dataConfig {
  string data_path = "data/";
  string data_name = "2011_09_26";
};

class dataHandler {
 public:
  explicit dataHandler(const dataConfig &data_config);
  ~dataHandler();

  // member functions
  string get_data_path();
  string get_data_name();
  string get_full_path();

 private:
  string data_path_, data_name_, full_path_;
};