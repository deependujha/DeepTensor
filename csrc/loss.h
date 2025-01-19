#pragma once
#include <memory>
#include "tensor.h"
#include "value.h"

std::shared_ptr<Value> mean_squared_error(
    std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> y);

std::shared_ptr<Value> cross_entropy(
    std::shared_ptr<Tensor> logits,
    int actualIdx);

std::shared_ptr<Value> binary_cross_entropy(
    std::shared_ptr<Tensor> logits,
    int actualIdx);
