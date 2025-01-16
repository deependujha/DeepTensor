#pragma once
#include <memory>
#include "../neural_network.h"
#include "../tensor.h"
#include "../utils.h"

class Conv2D : public Layer {
private:
  int in_channels;
  int out_channels;
  int kernel_size;
  int stride;
  int padding;
  int seed = -1;
  std::string technique = constant::HE;
  std::string mode = constant::NORMAL;
  std::shared_ptr<Tensor>
      weights; // Shape: [out_channels, in_channels, kernel_size, kernel_size]
  std::shared_ptr<Tensor> bias; // Shape: [out_channels]

  void _initialize() {
    // Initialize weights and bias
    this->weights = std::make_shared<Tensor>(
        std::vector<int>{out_channels, in_channels, kernel_size, kernel_size});
    this->bias = std::make_shared<Tensor>(std::vector<int>{out_channels});

    // Determine the seed to use
    int seed_to_use = (this->seed == -1) ? 42 : this->seed;

    // Create the RandomNumberGenerator
    RandomNumberGenerator rng(
        this->technique, this->mode, this->in_channels, this->out_channels, seed_to_use);
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
          for (int kw = 0; kw < kernel_size; ++kw) {
            double weight = rng.generate();
            weights->set({oc, ic, kh, kw}, std::make_shared<Value>(weight));
          }
        }
      }
      bias->set(oc, std::make_shared<Value>(0.0));
    }
  }

public:
  Conv2D(
      int in_channels,
      int out_channels,
      int kernel_size,
      int stride = 1,
      int padding = 0)
      : in_channels(in_channels),
        out_channels(out_channels),
        kernel_size(kernel_size),
        stride(stride),
        padding(padding) {
    _initialize();
  }
  Conv2D(
      int in_channels,
      int out_channels,
      int kernel_size,
      int stride,
      int padding,
      int seed,
      const std::string& technique,
      const std::string& mode)
      : in_channels(in_channels),
        out_channels(out_channels),
        kernel_size(kernel_size),
        stride(stride),
        padding(padding) {
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
    this->seed = seed;
    this->technique = technique;
    this->mode = mode;
    _initialize();
  }

  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    auto input_shape = input->shape; // [batch_size, in_channels, height, width]
    int batch_size = input_shape[0];
    int height = input_shape[2];
    int width = input_shape[3];

    // Compute output dimensions
    int output_height = (height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width - kernel_size + 2 * padding) / stride + 1;

    // Output tensor
    auto output = std::make_shared<Tensor>(std::vector<int>{
        batch_size, out_channels, output_height, output_width});

    for (int b = 0; b < batch_size; ++b) {
      for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < output_height; ++oh) {
          for (int ow = 0; ow < output_width; ++ow) {
            // Compute the dot product of the kernel and the input patch
            double result = 0.0;
            for (int ic = 0; ic < in_channels; ++ic) {
              for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                  int ih = oh * stride + kh - padding;
                  int iw = ow * stride + kw - padding;
                  if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    result += input->get({b, ic, ih, iw})->data *
                        weights->get({oc, ic, kh, kw})->data;
                  }
                }
              }
            }
            result += bias->get(oc)->data; // Add bias
            output->set({b, oc, oh, ow}, std::make_shared<Value>(result));
          }
        }
      }
    }

    return output;
  }

  std::string printMe() override {
    return "Conv2D(in_channels=" + std::to_string(in_channels) +
        ", out_channels=" + std::to_string(out_channels) +
        ", kernel_size=" + std::to_string(kernel_size) +
        ", stride=" + std::to_string(stride) +
        ", padding=" + std::to_string(padding) + ")";
  }

  void zero_grad() override {
    weights->zero_grad();
    bias->zero_grad();
  }
};

class MaxPooling2D : public Layer {
private:
  int pool_size;
  int stride;

public:
  MaxPooling2D(int pool_size, int stride = 1)
      : pool_size(pool_size), stride(stride) {}

  std::shared_ptr<Tensor> call(std::shared_ptr<Tensor> input, bool using_cuda)
      override {
    auto input_shape = input->shape; // [batch_size, channels, height, width]
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];

    // Compute output dimensions
    int output_height = (height - pool_size) / stride + 1;
    int output_width = (width - pool_size) / stride + 1;

    // Output tensor
    auto output = std::make_shared<Tensor>(
        std::vector<int>{batch_size, channels, output_height, output_width});

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < output_height; ++oh) {
          for (int ow = 0; ow < output_width; ++ow) {
            std::shared_ptr<Value> max_val = std::make_shared<Value>(
                -std::numeric_limits<double>::infinity());
            for (int ph = 0; ph < pool_size; ++ph) {
              for (int pw = 0; pw < pool_size; ++pw) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                if (ih < height && iw < width) {
                  std::shared_ptr<Value> curr_val = input->get({b, c, ih, iw});
                  if (max_val->data < curr_val->data) {
                    max_val = curr_val;
                  }
                }
              }
            }
            output->set({b, c, oh, ow}, max_val);
          }
        }
      }
    }

    return output;
  }

  std::string printMe() override {
    return "MaxPooling2D(pool_size=" + std::to_string(pool_size) +
        ", stride=" + std::to_string(stride) + ")";
  }

  void zero_grad() override {
    // No trainable parameters, so no action needed
  }
};
