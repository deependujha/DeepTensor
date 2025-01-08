#pragma once
#include <string>
#include "../neural_network.h"

class ReLu : public Layer {
public:
  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    return input->relu();
  }

  std::string printMe() override {
    return "ReLu()";
  }

  void zero_grad() override {};

  std::vector<std::shared_ptr<Value>> parameters() override {
    // no parameters
    return std::vector<std::shared_ptr<Value>>{};
  }
};

class GeLu : public Layer {
public:
  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    return input->gelu();
  }

  std::string printMe() override {
    return "GeLu()";
  }

  void zero_grad() override {};

  std::vector<std::shared_ptr<Value>> parameters() override {
    // no parameters
    return std::vector<std::shared_ptr<Value>>{};
  }
};

class Tanh : public Layer {
public:
  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    return input->tanh();
  }

  std::string printMe() override {
    return "Tanh()";
  }

  void zero_grad() override {};

  std::vector<std::shared_ptr<Value>> parameters() override {
    // no parameters
    return std::vector<std::shared_ptr<Value>>{};
  }
};

class Sigmoid : public Layer {
public:
  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    return input->sigmoid();
  }

  std::string printMe() override {
    return "Sigmoid()";
  }

  void zero_grad() override {};

  std::vector<std::shared_ptr<Value>> parameters() override {
    // no parameters
    return std::vector<std::shared_ptr<Value>>{};
  }
};

class LeakyReLu : public Layer {
public:
  double alpha;
  LeakyReLu(double alpha) : alpha(alpha) {}
  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    return input->leakyRelu(this->alpha);
  }

  std::string printMe() override {
    return "LeakyReLu(" + std::to_string(this->alpha) + ")";
  }

  void zero_grad() override {};

  std::vector<std::shared_ptr<Value>> parameters() override {
    // no parameters
    return std::vector<std::shared_ptr<Value>>{};
  }
};

class SoftMax : public Layer {
public:
  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    return input->softmax();
  }

  std::string printMe() override {
    return "SoftMax()";
  }

  void zero_grad() override {};

  std::vector<std::shared_ptr<Value>> parameters() override {
    // no parameters
    return std::vector<std::shared_ptr<Value>>{};
  }
};
