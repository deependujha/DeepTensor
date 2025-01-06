#pragma once
#include "json/json.h"

void write_to_file(std::string filename);

Json::Value read_json_file(const std::string& filename);
