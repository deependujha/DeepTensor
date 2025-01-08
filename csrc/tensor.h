#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "value.h"

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  std::vector<int> shape;
  std::vector<int> strides; // jump each index needs to make
  std::vector<std::shared_ptr<Value>> v;
  int maxIdx;
  int minIdx;

  Tensor(std::vector<int> shape) : shape(std::move(shape)) {
    int total_size = 1;
    for (auto& e : this->shape) {
      total_size *= e;
    }
    v.resize(total_size);

    strides.resize(this->shape.size());
    strides.back() = 1;
    for (int i = int(this->shape.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * this->shape[i + 1];
    }

    this->minIdx = 0;
    this->maxIdx = 1;
    for (auto& e : this->shape) {
      this->maxIdx *= e;
    }
    maxIdx--; // 1 less
  }

  std::string tensor_shape_str() {
    std::string shape_str = "(";
    for (auto& e : this->shape) {
      shape_str += std::to_string(e) + ", ";
    }
    shape_str += ")";
    return shape_str;
  }

  void set(std::vector<int> idx, std::shared_ptr<Value> _v) {
    int original_idx = normalize_idx(idx);
    if ((original_idx < this->minIdx) || (original_idx > this->maxIdx)) {
      std::string error_msg = "Index must be in the range. Limit (" +
          std::to_string(this->minIdx) + "," + std::to_string(this->maxIdx) +
          "), but found: " + std::to_string(original_idx) + ".";

      throw std::runtime_error(error_msg);
    }
    this->v[original_idx] = _v;
  }

  std::shared_ptr<Value> get(std::vector<int> idx) {
    int original_idx = normalize_idx(idx);
    if ((original_idx < this->minIdx) || (original_idx > this->maxIdx)) {
      std::string error_msg = "Index must be in the range. Limit (" +
          std::to_string(this->minIdx) + "," + std::to_string(this->maxIdx) +
          "), but found: " + std::to_string(original_idx) + ".";

      throw std::runtime_error(error_msg);
    }
    return this->v[original_idx];
  }

  // real index
  void set(int idx, std::shared_ptr<Value> _v) {
    if ((idx < this->minIdx) || (idx > this->maxIdx)) {
      std::string error_msg = "Index must be in the range. Limit (" +
          std::to_string(this->minIdx) + "," + std::to_string(this->maxIdx) +
          "), but found: " + std::to_string(idx) + ".";

      throw std::runtime_error(error_msg);
    }
    this->v[idx] = _v;
  }

  // real index
  std::shared_ptr<Value> get(int idx) {
    if ((idx < this->minIdx) || (idx > this->maxIdx)) {
      std::string error_msg = "Index must be in the range. Limit (" +
          std::to_string(this->minIdx) + "," + std::to_string(this->maxIdx) +
          "), but found: " + std::to_string(idx) + ".";

      throw std::runtime_error(error_msg);
    }
    return this->v[idx];
  }

  unsigned dims() {
    return this->shape.size();
  }

  int normalize_idx(std::vector<int> idx);

  // tensor specific operations (so layers can directly call them)
  void zero_grad() {
    for (auto& e : this->v) {
      e->grad = 0;
    }
  }

  void backward() {
    for (auto& e : this->v) {
      e->backward();
    }
  }

  void remove_redundant_rows(std::shared_ptr<Tensor> t) {
    // remove redundant rows(1)
    std::vector<int> new_shape = {};
    for (auto& e : t->shape) {
      if (e > 1) {
        new_shape.push_back(e);
      }
    }
    if (new_shape.size() == 0) {
      new_shape.push_back(1);
    }
    t->shape = new_shape;
  }

  std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other) {
    remove_redundant_rows(shared_from_this());
    remove_redundant_rows(other);

    if (this->shape != other->shape) {
      std::string this_shape_str = "(";
      for (auto& e : this->shape) {
        this_shape_str += std::to_string(e) + ", ";
      }
      this_shape_str += ")";

      std::string other_shape_str = "(";
      for (auto& e : other->shape) {
        other_shape_str += std::to_string(e) + ", ";
      }
      other_shape_str += ")";

      throw std::runtime_error(
          "Tensors must have the same shape for addition. Got shapes: " +
          this_shape_str + " and " + other_shape_str);
    }

    std::shared_ptr<Tensor> out = std::make_shared<Tensor>(other->shape);

    for (int i = 0; i < this->v.size(); i++) {
      std::shared_ptr<Value> curr_v = this->get(i)->add(other->get(i));
      out->set(i, std::move(curr_v));
    }
    return out;
  }

  std::shared_ptr<Tensor> div(std::shared_ptr<Value> other) {
    if (other->data == 0) {
      throw std::runtime_error("Division is not supported by Value(0)");
    }

    std::shared_ptr<Tensor> out = std::make_shared<Tensor>(this->shape);

    for (int i = 0; i < this->v.size(); i++) {
      std::shared_ptr<Value> curr_v = this->get(i)->div(other);
      out->set(i, std::move(curr_v));
    }
    return out;
  }

  std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other) {
    if (!other) {
      throw std::runtime_error("Cannot perform matmul with a null tensor.");
    }

    // Determine effective shapes
    std::vector<int> this_shape = this->shape;
    std::vector<int> other_shape = other->shape;

    // Reshape if either is a vector (1D tensor)
    if (this_shape.size() == 1) {
      this_shape.insert(this_shape.begin(), 1); // Treat as row vector
      std::vector<int> new_shape = {1, this_shape[0]};
      this->shape = new_shape;
    }
    if (other_shape.size() == 1) {
      other_shape.push_back(1); // Treat as column vector
      other->shape.push_back(1);
    }

    // Validate dimensions for matrix multiplication
    if (this_shape[1] != other_shape[0]) {
      throw std::runtime_error(
          "Dimensions do not align for matmul. Got shapes: (" +
          std::to_string(this_shape[0]) + ", " + std::to_string(this_shape[1]) +
          ") and (" + std::to_string(other_shape[0]) + ", " +
          std::to_string(other_shape[1]) + ")");
    }

    // Compute output shape
    std::vector<int> output_shape = {this_shape[0], other_shape[1]};
    auto out = std::make_shared<Tensor>(output_shape);

    // Perform matrix multiplication
    for (int i = 0; i < output_shape[0]; i++) {
      for (int j = 0; j < output_shape[1]; j++) {
        std::shared_ptr<Value> sum = std::make_shared<Value>(0);
        for (int k = 0; k < this_shape[1]; k++) {
          sum = sum->add(this->get({i, k})->mul(other->get({k, j})));
        }
        out->set({i, j}, sum);
      }
    }

    return out;
  }

  // non-linear layers in tesor
  std::shared_ptr<Tensor> relu() {
    std::shared_ptr<Tensor> out = std::make_shared<Tensor>(this->shape);
    int i = 0;
    for (auto& e : this->v) {
      std::shared_ptr<Value> curr = e->relu();
      out->set(i, curr);
      i++;
    }
    return out;
  }

  std::shared_ptr<Tensor> tanh() {
    std::shared_ptr<Tensor> out = std::make_shared<Tensor>(this->shape);
    int i = 0;
    for (auto& e : this->v) {
      std::shared_ptr<Value> curr = e->tanh();
      out->set(i, curr);
      i++;
    }
    return out;
  }

  std::shared_ptr<Tensor> gelu() {
    std::shared_ptr<Tensor> out = std::make_shared<Tensor>(this->shape);
    int i = 0;
    for (auto& e : this->v) {
      std::shared_ptr<Value> curr = e->gelu();
      out->set(i, curr);
      i++;
    }
    return out;
  }

  std::shared_ptr<Tensor> sigmoid() {
    std::shared_ptr<Tensor> out = std::make_shared<Tensor>(this->shape);
    int i = 0;
    for (auto& e : this->v) {
      std::shared_ptr<Value> curr = e->sigmoid();
      out->set(i, curr);
      i++;
    }
    return out;
  }

  std::shared_ptr<Tensor> leakyRelu(double alpha) {
    std::shared_ptr<Tensor> out = std::make_shared<Tensor>(this->shape);
    int i = 0;
    for (auto& e : this->v) {
      std::shared_ptr<Value> curr = e->leakyRelu(alpha);
      out->set(i, curr);
      i++;
    }
    return out;
  }

  std::shared_ptr<Tensor> softmax() {
    // Step 1: Find the maximum value for numerical stability
    auto max_val = *std::max_element(
        this->v.begin(),
        this->v.end(),
        [](const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
          return a->data < b->data;
        });
    // Step 2: Compute exp(x_i - max_val) for each input
    std::shared_ptr<Tensor> exp_vals = std::make_shared<Tensor>(this->shape);

    int i = 0;
    for (auto& val : this->v) {
      auto curr_exp_val = val->sub(max_val)->exp();
      exp_vals->set(i, curr_exp_val);
      i++;
    }

    // Step 3: Compute the sum of exp(x_i - max_val)
    std::shared_ptr<Value> sum_exp = std::make_shared<Value>(0.0);
    for (int i = 0; i <= exp_vals->maxIdx; i++) {
      sum_exp = sum_exp->add(exp_vals->get(i));
    }

    // Step 4: Compute softmax = exp(x_i - max_val) / sum_exp
    std::shared_ptr<Tensor> softmax_vals =
        std::make_shared<Tensor>(this->shape);

    for (int i = 0; i <= softmax_vals->maxIdx; i++) {
      softmax_vals->set(i, exp_vals->get(i)->div(sum_exp));
    }

    return softmax_vals;
  }
};
