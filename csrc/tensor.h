#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "value.h"

class Tensor {
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
      e->executeBackWardMethod();
    }
  }

  std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other) {
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
    }
    if (other_shape.size() == 1) {
      other_shape.push_back(1); // Treat as column vector
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
};
