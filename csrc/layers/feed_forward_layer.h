#pragma once
#include <memory>
#include <string>
#include "../constant.h"
#include "../neural_network.h"
#include "../tensor.h"
#include "../utils.h"

class FeedForwardLayer : public Layer {
private:
  int nin; // no_of_inputs
  int nout; // no_of_outputs
  int seed = -1;
  std::shared_ptr<Tensor> weights; // nin * nout (nout rows of nin values)
  std::shared_ptr<Tensor> bias; // nin * nout (nout rows of nin values)
  std::string technique = constant::HE;
  std::string mode = constant::NORMAL;

  void _initialize() {
    this->weights =
        std::make_shared<Tensor>(std::vector<int>{this->nin, this->nout});
    this->bias = std::make_shared<Tensor>(std::vector<int>{this->nout});

    // Determine the seed to use
    int seed_to_use = (this->seed == -1) ? 42 : this->seed;

    // Create the RandomNumberGenerator
    RandomNumberGenerator rng(
        this->technique, this->mode, this->nin, this->nout, seed_to_use);

    for (int i = 0; i < this->nin; i++) {
      for (int j = 0; j < this->nout; j++) {
        double data = rng.generate();
        std::shared_ptr<Value> curr_v = std::make_shared<Value>(data);

        this->weights->set({i, j}, curr_v);
      }
    }
    for (int j = 0; j < this->nout; j++) {
      this->bias->set(j, std::make_shared<Value>(0));
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
  FeedForwardLayer(
      int nin,
      int nout,
      int seed,
      const std::string& technique,
      const std::string& mode)
      : nin(nin), nout(nout), seed(seed) {
    if (technique != constant::HE && technique != constant::XAVIER) {
      throw std::runtime_error(
          "FeedForward layer expects 'technique' to be either 'XAVIER' or 'HE'. Got: " +
          technique);
    }
    if (mode != constant::UNIFORM && mode != constant::NORMAL) {
      throw std::runtime_error(
          "FeedForward layer expects 'mode' to be either 'UNIFORM' or 'NORMAL'. Got: " +
          mode);
    }
    this->technique = technique;
    this->mode = mode;
    _initialize();
  }

  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    if (input->shape[0] != this->nin) {
      std::string error_msg =
          "Input tensor shape mismatch with layer's weights. Expected input size: " +
          std::to_string(this->nin) +
          ", but got input of size: " + std::to_string(input->shape[0]);
      throw std::invalid_argument(error_msg);
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

  std::vector<std::shared_ptr<Value>> parameters() override {
    std::vector<std::shared_ptr<Value>> out;
    for (int i = 0; i <= this->weights->maxIdx; i++) {
      out.push_back(this->weights->get(i));
    }
    for (int i = 0; i <= this->bias->maxIdx; i++) {
      out.push_back(this->bias->get(i));
    }
    return out;
  }
};
