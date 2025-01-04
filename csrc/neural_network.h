#pragma once
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include "utils.h"
#include "value.h"

class Module {
public:
  virtual ~Module() = default;
  virtual std::vector<std::shared_ptr<Value>> parameters();

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
    assert(input.size() == this->nin);

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

class Layer : public Module {
private:
  int nin; // no_of_inputs
  int nout; // no_of_outputs
  bool nonlin = false;
  int seed = 42;
  std::vector<std::shared_ptr<Neuron>> neurons;

  void _initialize() {
    for (int i = 0; i < this->nout; i++) {
      std::shared_ptr<Neuron> tmp_n =
          std::make_shared<Neuron>(this->nin, this->nonlin, this->seed);
      neurons.push_back(tmp_n);
    }
  }

public:
  Layer(int nin, int nout) : nin(nin), nout(nout) {
    _initialize();
  }
  Layer(int nin, int nout, bool nonlin) : nin(nin), nout(nout), nonlin(nonlin) {
    _initialize();
  }
  Layer(int nin, int nout, bool nonlin, int seed)
      : nin(nin), nout(nout), nonlin(nonlin), seed(seed) {
    _initialize();
  }

  std::vector<std::shared_ptr<Value>> call(
      std::vector<std::shared_ptr<Value>> input) {
    std::vector<std::shared_ptr<Value>> out;
    for (int i = 0; i < this->nout; i++) {
      std::shared_ptr<Value> tmp = this->neurons[i]->call(input);
      out.push_back(tmp);
    }
    return out;
  }

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> p;
    for (auto& e : neurons) {
      auto _ep = e->parameters();
      p.insert(p.end(), _ep.begin(), _ep.end());
    }
    return p;
  }

  std::string printMe() {
    std::string s = "Layer of [";
    for (auto& e : this->neurons) {
      s += e->printMe();
      s += ", ";
    }
    s += "]";
    return s;
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

    for (int i = 1; i < nouts.size(); i++) {
      std::shared_ptr<Layer> _l = std::make_shared<Layer>(
          this->nouts[i - 1], this->nouts[i], this->nonlin, this->seed);
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
    std::string s = "Layer of [";
    for (auto& e : this->layers) {
      s += e->printMe();
      s += ", ";
    }
    s += "]";
    return s;
  }
};
