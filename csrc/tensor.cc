#include "tensor.h"
#include <cassert>
#include <stdexcept>
#include <string>

// ==== my lightning ai interview question (with Luca Antiga, CTO, Lightning AI)
// that I miserably failed. :(

/// convert multi-dimensional index to one-dimensinal index
/// for an array of 1 dim and shape: (5) => [3] = 0+3
/// for an array of 2 dim and shape: (5,4) => [3,2] = 14
/// for an array of 3 dim and shape (3,3,2) => [0,0,0] = 0
///                                  (3,3,2) => [0,0,1] = 1
///                                  (3,3,2) => [0,1,0] = 2
///                                  (3,3,2) => [0,1,1] = 3
///                                  (3,3,2) => [0,2,0] = 4
///                                  (3,3,2) => [0,2,1] = 5
///                                  (3,3,2) => [1,0,0] = 6
int Tensor::normalize_idx(std::vector<int> idx) {
  if (idx.size() != this->shape.size()) {
    std::string shape_str = "";
    for (auto& e : this->shape) {
      shape_str += std::to_string(e) + ",";
    }

    std::string idx_str = "";
    for (auto& e : idx) {
      idx_str += std::to_string(e) + ",";
    }
    std::string error_string =
        "idx and tensor's shape don't have similar dims: tensor shape (" +
        shape_str + ") vs idx shape(" + idx_str + ")\n";
    throw std::runtime_error(error_string);
  }

  int final_idx = 0;

  for (int i = 0; i < int(this->shape.size()); i++) {
    // int _curr_pro = 1;
    // for (int j = i + 1; j < int(this->shape.size()); j++) {
    //   _curr_pro *= this->shape[j];
    // }
    // _curr_pro *= idx[i];

    // simply use strides instead of recomputing each time
    int _curr_pro = idx[i] * this->strides[i];
    final_idx += _curr_pro;
  }
  return final_idx;
}
