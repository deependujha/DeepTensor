#pragma once
#include <string>

namespace constant {

// For weight initialization
std::string XAVIER = "XAVIER"; // good for sigmoid or tanh activation functions
std::string HE = "HE"; // good for relu and its variants
std::string NORMAL = "NORMAL";
std::string UNIFORM = "UNIFORM";

} // namespace constant
