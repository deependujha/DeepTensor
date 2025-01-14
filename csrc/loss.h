#pragma once
#include "value.h"
#include "tensor.h"
#include <memory>

std::shared_ptr<Value> mean_squared_error(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y);

std::shared_ptr<Value> cross_entropy(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> actual);
