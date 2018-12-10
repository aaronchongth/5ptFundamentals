#pragma once

#include <stdio.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

// opencv
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/sfm.hpp"
#include "opencv2/xfeatures2d.hpp"

// custom libraries
#include "5pt_algo/5pt.hpp"
#include "7pt_algo/7pt.hpp"
#include "data_handler/data_handler.hpp"
#include "extra_data_handler/extra_data_handler.hpp"
#include "utilities/utilities.hpp"

inline void cry(std::string error_msg) {
  std::ostringstream err;
  err << error_msg;
  throw std::runtime_error(err.str().c_str());
}