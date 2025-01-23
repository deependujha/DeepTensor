#pragma once
#include <memory>
#include <string>
#include "../neural_network.h"
#include "../tensor.h"

class Flatten : public Layer {
public:
  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    return input->flatten();
  }

  std::string printMe() override {
    return "Flatten()";
  }

  void zero_grad() override {};
};
