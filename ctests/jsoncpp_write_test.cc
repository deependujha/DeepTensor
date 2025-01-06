#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "jsoncpp.h"

namespace fs = std::filesystem;

// Test fixture for setting up a temporary file and MLP model
class JsonCppIOTest : public ::testing::Test {
protected:
  std::string temp_filename;

  void SetUp() override {
    // Temporary filename
    temp_filename = "temp_model.json";
  }

  void TearDown() override {
    // Remove temporary file after test
    if (fs::exists(temp_filename)) {
      fs::remove(temp_filename);
    }
  }
};

// Test for saving the model to a file
TEST_F(JsonCppIOTest, WritesJsonFile) {
  // Call write_to_file function
  write_to_file(temp_filename);

  // Check if the file was created
  ASSERT_TRUE(fs::exists(temp_filename))
      << "Save function did not create a file.";

  // Verify content (optional: parse and check JSON structure)
  std::ifstream file(temp_filename);
  ASSERT_TRUE(file.is_open()) << "File was not opened successfully.";

  Json::Value json_model;
  file >> json_model;

  // Check for specific values
  ASSERT_EQ(json_model["action"], "run");
  ASSERT_EQ(json_model["data"]["number"], 1);
}
