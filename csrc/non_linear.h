#pragma once
#include <memory>
#include <vector>
#include "value.h"

// non-linear layers
namespace NonLinear {
std::vector<std::shared_ptr<Value>> relu(
    std::vector<std::shared_ptr<Value>> inp);

std::vector<std::shared_ptr<Value>> tanh(
    std::vector<std::shared_ptr<Value>> inp);

std::vector<std::shared_ptr<Value>> gelu(
    std::vector<std::shared_ptr<Value>> inp);

std::vector<std::shared_ptr<Value>> sigmoid(
    std::vector<std::shared_ptr<Value>> inp);

std::vector<std::shared_ptr<Value>> leakyRelu(
    double alpha,
    std::vector<std::shared_ptr<Value>> inp);

std::vector<std::shared_ptr<Value>> softmax(
    std::vector<std::shared_ptr<Value>> inp);
} // namespace NonLinear
