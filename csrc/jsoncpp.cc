#include "jsoncpp.h"
#include <json/json.h>
#include <fstream>
#include <stdexcept>

void write_to_file(std::string filename) {
  Json::Value root;
  Json::Value data;
  root["action"] = "run";
  data["number"] = 1;
  root["data"] = data;

  Json::StreamWriterBuilder builder;
  const std::string json_file = Json::writeString(builder, root);
  //   std::cout << json_file << std::endl;
  std::ofstream file(filename);
  file << json_file;
  file.close();
}

Json::Value read_json_file(const std::string& filename) {
  Json::Value root;
  std::ifstream ifs(filename, std::ios::binary);

  // Check if the file was successfully opened
  if (!ifs.is_open()) {
    throw std::runtime_error("Error: Unable to open file " + filename);
  }

  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  JSONCPP_STRING errs;

  // Parse the JSON
  if (!Json::parseFromStream(builder, ifs, &root, &errs)) {
    throw std::runtime_error("Error parsing JSON file: " + errs);
  }

  return root;
}
