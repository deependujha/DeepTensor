#pragma once
#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensor.h"

class Layer {
public:
  virtual ~Layer() = default;

  virtual std::shared_ptr<Tensor> call(
      std::shared_ptr<Tensor> input,
      bool using_cuda) = 0;

  virtual std::string printMe() = 0;

  virtual std::vector<std::shared_ptr<Value>> parameters() {
    // no parameters
    return std::vector<std::shared_ptr<Value>>{};
  }

  virtual void zero_grad() = 0;
};

class Model {
public:
  bool using_cuda = false;
  std::vector<std::shared_ptr<Layer>> layers;

  Model(std::vector<std::shared_ptr<Layer>> layers, bool using_cuda)
      : layers(std::move(layers)), using_cuda(using_cuda) {}

  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> out = input;
    for (auto& e : this->layers) {
      out = e->call(out, this->using_cuda);
    }
    return out;
  }

  void zero_grad() {
    for (auto& e : this->layers) {
      e->zero_grad();
    }
  }

  std::string printMe() {
    std::string s = "Model(\n";
    for (auto& e : this->layers) {
      s += "\t";
      s += e->printMe();
      s += ",\n";
    }
    s += ")";
    return s;
  }

  std::vector<std::shared_ptr<Value>> parameters() {
    std::vector<std::shared_ptr<Value>> out;
    for (auto& e : this->layers) {
      std::vector<std::shared_ptr<Value>> curr = e->parameters();
      out.insert(out.end(), curr.begin(), curr.end());
    }
    return out;
  }

  void save_model(std::string filename) {}

  void load_model(std::string filename) {}
};
