#pragma once
#include "neural_network.h"

class LinearLayer : public Layer {
private:
  int nin; // no_of_inputs
  int nout; // no_of_outputs
  bool nonlin = false;
  int seed = 42;
  std::vector<std::shared_ptr<Neuron>> neurons;

  void _initialize() {
    for (int i = 0; i < this->nout; i++) {
      std::shared_ptr<Neuron> tmp_n = std::make_shared<Neuron>(
          this->nin, this->nonlin, this->seed + (1000 * i));
      neurons.push_back(tmp_n);
    }
  }

public:
  LinearLayer(int nin, int nout) : nin(nin), nout(nout) {
    _initialize();
  }
  LinearLayer(int nin, int nout, bool nonlin) : nin(nin), nout(nout), nonlin(nonlin) {
    _initialize();
  }
  LinearLayer(int nin, int nout, bool nonlin, int seed)
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
    std::string s = "Layer(" + std::to_string(this->nin) + "," +
        std::to_string(this->nout) + ")";
    return s;
  }
};
