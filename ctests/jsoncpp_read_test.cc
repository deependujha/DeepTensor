#include <gtest/gtest.h>
#include <fstream>
#include "jsoncpp.h"

// Test fixture to manage temporary files
class ReadJsonFileTest : public ::testing::Test {
protected:
  std::string temp_valid_file = "valid.json";
  std::string temp_invalid_file = "invalid.json";

  void SetUp() override {
    // Create a valid JSON file
    std::ofstream valid_file(temp_valid_file);
    valid_file << R"({
            "name": "DeepTensor",
            "version": 1.0,
            "layers": [2, 3, 1]
        })";
    valid_file.close();

    // Create an invalid JSON file
    std::ofstream invalid_file(temp_invalid_file);
    invalid_file << R"({
            "name": "Invalid JSON",
            "version": 1.0,
            "layers": [2, 3, 1)";
    invalid_file.close();
  }

  void TearDown() override {
    // Remove temporary files
    std::remove(temp_valid_file.c_str());
    std::remove(temp_invalid_file.c_str());
  }
};

// Test: Successfully reads a valid JSON file
TEST_F(ReadJsonFileTest, ReadValidJsonFile) {
  Json::Value json = read_json_file(temp_valid_file);
  ASSERT_EQ(json["name"].asString(), "DeepTensor");
  ASSERT_FLOAT_EQ(json["version"].asFloat(), 1.0);
  ASSERT_EQ(json["layers"].size(), 3);
  ASSERT_EQ(json["layers"][0].asInt(), 2);
}

// Test: Throws an error when the file does not exist
TEST_F(ReadJsonFileTest, FileNotFound) {
  ASSERT_THROW(read_json_file("non_existent.json"), std::runtime_error);
}

// Test: Throws an error when the file contains invalid JSON
TEST_F(ReadJsonFileTest, InvalidJsonFile) {
  ASSERT_THROW(read_json_file(temp_invalid_file), std::runtime_error);
}
