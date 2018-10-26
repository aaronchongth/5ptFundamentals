#include "data_handler.hpp"

using namespace std;

dataHandler::dataHandler(const dataConfig& data_config)
    : data_path_(data_config.data_path), data_name_(data_config.data_name) {
  cout << "init done." << endl;
}

dataHandler::~dataHandler() { cout << "destructor called." << endl; }

string dataHandler::get_full_path() {
  cout << data_path_ << endl;
  cout << data_name_ << endl;

  string full_path = data_path_ + "/" + data_name_;
  cout << full_path << endl;
  return full_path;
}