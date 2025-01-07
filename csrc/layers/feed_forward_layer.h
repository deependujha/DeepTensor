#pragma once
#include <memory>
#include "../neural_network.h"
#include "../tensor.h"
#include "../utils.h"

class FeedForwardLayer : public Layer {
private:
  int nin; // no_of_inputs
  int nout; // no_of_outputs
  int seed = 42;
  std::shared_ptr<Tensor> weights; // nin * nout (nout rows of nin values)
  std::shared_ptr<Tensor> bias; // nin * nout (nout rows of nin values)

  void _initialize() {
    this->weights =
        std::make_shared<Tensor>(std::vector<int>{this->nin, this->nout});
    this->bias = std::make_shared<Tensor>(std::vector<int>{this->nout});

    // Create two instances of RandomNumberGenerator with the same seed
    RandomNumberGenerator rng(this->seed);

    for (int i = 0; i < this->nout; i++) {
      for (int j = 0; j < this->nin; j++) {
        double data = rng.generate();
        std::shared_ptr<Value> curr_v = std::make_shared<Value>(data);

        this->weights->set({i, j}, curr_v);
      }
      this->bias->set(i, std::make_shared<Value>(0));
    }
  }

public:
  FeedForwardLayer(int nin, int nout) : nin(nin), nout(nout) {
    _initialize();
  }
  FeedForwardLayer(int nin, int nout, int seed)
      : nin(nin), nout(nout), seed(seed) {
    _initialize();
  }

  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    if (input->shape[0] != this->nin) {
      throw std::invalid_argument(
          "Input tensor shape mismatch with layer's weights.");
    }
    std::shared_ptr<Tensor> out = input->matmul(this->weights)->add(this->bias);
    return out;
  }

  void zero_grad() override {
    this->weights->zero_grad();
    this->bias->zero_grad();
  }

  std::string printMe() override {
    std::string s = "Layer(" + std::to_string(this->nin) + "," +
        std::to_string(this->nout) + ")";
    return s;
  }
};
