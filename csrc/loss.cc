#include "loss.h"
#include <memory>
#include <stdexcept>
#include <string>
#include "value.h"

std::shared_ptr<Value> mean_squared_error(
    std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> y) {
  if (x->shape != y->shape) {
    std::string x_shape_str = x->tensor_shape_str();
    std::string y_shape_str = y->tensor_shape_str();
    std::string error_string =
        "Shapes of the two tensors for computing MSE don't match: tensor-1 shape (" +
        x_shape_str + ") vs tensor-1 shape(" + y_shape_str + ")\n";
    throw std::runtime_error(error_string);
  }
  std::shared_ptr<Value> out = std::make_shared<Value>(0.0);
  int n = x->maxIdx;

  for (int i = 0; i <= n; i++) {
    std::shared_ptr<Value> diff = x->get(i)->sub(y->get(i));
    std::shared_ptr<Value> diff_squared = diff->pow(2);
    out = out->add(diff_squared);
  }
  return out->div(n + 1);
}

std::shared_ptr<Value> cross_entropy(
    std::shared_ptr<Tensor> logits,
    int actualIdx) {
  if (actualIdx < 0) {
    throw std::runtime_error(
        "Expected Idx can't be smaller than 0. Got: " +
        std::to_string(actualIdx));
  }
  if (logits->shape.size() != 1 || logits->shape[0] < actualIdx) {
    throw std::runtime_error(
        "logits must be a one-dimensional tensor. And actualIdx must be smaller than logits size. Got: logits shape =>" +
        logits->tensor_shape_str() +
        ", and expectedIdx: " + std::to_string(actualIdx));
  }
  // compute softmax of logits
  std::shared_ptr<Tensor> logits_softmax = logits->softmax();

  constexpr double EPSILION = 1e-6;

  for (int z = 0; z < logits_softmax->maxIdx; z++) {
    if (logits_softmax->get(z)->data <= 0.0) {
      logits_softmax->set(
          z, std::make_shared<Value>(EPSILION)); // Handle near-zero values
    } else if (logits_softmax->get(z)->data >= 1.0) {
      logits_softmax->set(
          z, std::make_shared<Value>(1.0 - EPSILION)); // Handle near-one values
    }
  }

  std::shared_ptr logits_ln = logits_softmax->get(actualIdx)->ln();

  return logits_ln->mul(-1); // not averaging it
}

std::shared_ptr<Value> binary_cross_entropy(
    std::shared_ptr<Tensor> logits,
    int actualIdx) {
  if (actualIdx < 0 || actualIdx > 1) {
    throw std::runtime_error(
        "Expected Idx can't be smaller than 0 or greater than 1. Got: " +
        std::to_string(actualIdx));
  }
  if (logits->shape.size() != 1) {
    throw std::runtime_error(
        "logits must be a one-dimensional tensor.. Got: logits shape =>" +
        logits->tensor_shape_str());
  }
  std::shared_ptr<Value> logit_value = logits->get(0);

  std::shared_ptr<Value> updated_logit_value = logit_value;
  if (actualIdx == 0) {
    updated_logit_value = std::make_shared<Value>(1.0)->sub(logit_value);
  }

  if (updated_logit_value->data < 0 || updated_logit_value->data > 1) {
    throw std::runtime_error(
        "logit value can't be less than 0, and more than 1. Got: " +
        std::to_string(logit_value->data));
  }

  constexpr double EPSILION = 1e-6;
  if (updated_logit_value->data <= 0.0) {
    updated_logit_value->data = EPSILION; // Handle near-zero values
  } else if (updated_logit_value->data >= 1.0) {
    updated_logit_value->data = 1.0 - EPSILION; // Handle near-one values
  }

  std::shared_ptr logits_ln = updated_logit_value->ln();
  return logits_ln->mul(-1);
}
