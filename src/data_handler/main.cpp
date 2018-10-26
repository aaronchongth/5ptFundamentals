#include "data_handler.hpp"

using namespace std;
using namespace cv;

int main() {
  dataConfig data_config;
  auto data_handler = dataHandler(data_config);

  Mat test_mat;

  cout << "All done." << endl;
}