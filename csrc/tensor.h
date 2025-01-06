#pragma once
#include <memory>
#include <vector>
#include "value.h"

class Tensor {
public:
  std::vector<int> shape;
  std::vector<int> strides; // jump each index needs to make
  std::vector<std::shared_ptr<Value>> v;

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
  }

  void set(std::vector<int> idx, std::shared_ptr<Value> _v) {
    int original_idx = normalize_idx(idx);
    this->v[original_idx] = _v;
  }

  std::shared_ptr<Value> get(std::vector<int> idx) {
    int original_idx = normalize_idx(idx);
    return this->v[original_idx];
  }

  unsigned dims() {
    return this->shape.size();
  }

  int normalize_idx(std::vector<int> idx);
};
