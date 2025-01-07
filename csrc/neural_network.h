#pragma once
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include "tensor.h"
#include "utils.h"
#include "value.h"

class Module {
public:
  virtual ~Module() = default;
  virtual std::vector<std::shared_ptr<Value>> parameters() = 0;

  void zero_grad() {
    std::vector<std::shared_ptr<Value>> p = parameters();

    for (auto& e : p) {
      e->grad = 0;
    }
  }
};

class Neuron : public Module {
  int nin;
  bool nonlin = false;
  std::vector<std::shared_ptr<Value>> weights;
  std::shared_ptr<Value> bias;

  void _initialize(int seed = 42) {
    // Create two instances of RandomNumberGenerator with the same seed
    RandomNumberGenerator rng(seed);

    for (int i = 0; i < this->nin; i++) {
      double data = rng.generate();
      std::shared_ptr<Value> w = std::make_shared<Value>(data);
      weights.push_back(w);
    }

    bias = std::make_shared<Value>(0);
  }

public:
  Neuron(int nin) : nin(nin) {
    _initialize();
  }
  Neuron(int nin, bool nonlin) : nin(nin), nonlin(nonlin) {
    _initialize();
  }
  Neuron(int nin, bool nonlin, int seed) : nin(nin), nonlin(nonlin) {
    _initialize(seed);
  }

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> p = this->weights;
    p.push_back(this->bias);
    return p;
  }

  std::shared_ptr<Value> call(std::vector<std::shared_ptr<Value>> input) {
    assert(int(input.size()) == this->nin);

    std::shared_ptr<Value> out = std::make_shared<Value>(0);

    for (int i = 0; i < this->nin; i++) {
      out = out->add(this->weights[i])->mul(input[i]); // out += w[i]*input[i]
    }
    out = out->add(this->bias);

    if (this->nonlin) {
      out = out->relu();
    }
    return out;
  }

  std::string printMe() {
    std::string s = "";
    if (nonlin) {
      s += "ReLU(nin=";
    } else {
      s += "Linear(nin=";
    }
    s += std::to_string(this->nin);
    s += ")";
    return s;
  }
};

class Layer {
public:
  virtual ~Layer() = default;

  virtual std::shared_ptr<Tensor> call(
      std::shared_ptr<Tensor> input,
      bool using_cuda) = 0;

  virtual std::vector<std::shared_ptr<Value>> parameters() = 0;

  virtual std::string printMe() = 0;

  void zero_grad() {
    std::vector<std::shared_ptr<Value>> p = this->parameters();

    for (auto& e : p) {
      e->grad = 0;
    }
  }
};

class MLP : public Module {
  int nin;
  std::vector<int> nouts;
  std::vector<std::shared_ptr<Layer>> layers;
  bool nonlin = false;
  int seed = 42;

  void _initialize() {
    // input to first-hidden layer
    std::shared_ptr<Layer> l1 = std::make_shared<Layer>(
        this->nin, this->nouts[0], this->nonlin, this->seed);
    this->layers.push_back(l1);

    for (int i = 1; i < int(nouts.size()); i++) {
      std::shared_ptr<Layer> _l = std::make_shared<Layer>(
          this->nouts[i - 1], this->nouts[i], this->nonlin, this->seed + i);
      this->layers.push_back(_l);
    }
  }

public:
  MLP(int nin, std::vector<int> nouts) : nin(nin), nouts(std::move(nouts)) {
    _initialize();
  }
  MLP(int nin, std::vector<int> nouts, bool nonlin)
      : nin(nin), nouts(std::move(nouts)), nonlin(nonlin) {
    _initialize();
  }
  MLP(int nin, std::vector<int> nouts, bool nonlin, int seed)
      : nin(nin), nouts(std::move(nouts)), nonlin(nonlin), seed(seed) {
    _initialize();
  }

  std::vector<std::shared_ptr<Value>> call(
      std::vector<std::shared_ptr<Value>> input) {
    std::vector<std::shared_ptr<Value>> out = input;
    for (auto& e : this->layers) {
      out = e->call(out);
    }
    return out;
  }

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> p;
    for (auto& e : this->layers) {
      auto _ep = e->parameters();
      p.insert(p.end(), _ep.begin(), _ep.end());
    }
    return p;
  }

  std::string printMe() {
    std::string s = "MLP of [";
    for (auto& e : this->layers) {
      s += e->printMe();
      s += ", ";
    }
    s += "]";
    return s;
  }
};

class Model : public Module {
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

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> p;
    for (auto& e : this->layers) {
      auto _ep = e->parameters();
      p.insert(p.end(), _ep.begin(), _ep.end());
    }
    return p;
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

  void save_model(std::string filename) {}

  void load_model(std::string filename) {}
};
