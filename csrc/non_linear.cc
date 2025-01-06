#include "non_linear.h"
#include <cmath>
#include <memory>
#include <unordered_set>
#include "value.h"

// non-linear methods for Value class
std::shared_ptr<Value> Value::relu() {
  double newData = this->data < 0 ? 0 : this->data;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, 'r');

  // Define the backward function
  std::function<void()> add_backward = [this, newVal]() {
    this->grad += newVal->grad * (newVal->data > 0 ? 1.0 : 0.0);
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::tanh() {
  // Approach -1 (using Value class primitives) (but creates unnecessary nodes)
  //   // Compute e^x and e^(-x) using the exp method
  //   std::shared_ptr<Value> exp_x = this->exp();
  //   std::shared_ptr<Value> exp_neg_x =
  //       this->exp()->rdiv(1.0); // e^(-x) = 1/(e^x) (e^x calculation is
  //       heavier than dividing from 1)

  //   // Compute numerator and denominator
  //   std::shared_ptr<Value> numerator = exp_x->sub(exp_neg_x);
  //   std::shared_ptr<Value> denominator = exp_x->add(exp_neg_x);

  //   // Compute tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  //   std::shared_ptr<Value> result = numerator->div(denominator);

  //   return result;

  // approach 2 - directly create resultant node and write the backprop
  // Forward pass: compute tanh(x)
  double tanhData = std::tanh(this->data);
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(tanhData, std::move(prev), 't');

  // Backward pass: gradient of tanh(x) is (1 - tanh^2(x))
  std::function<void()> add_backward = [this, newVal]() {
    this->grad += (1.0 - (newVal->data * newVal->data)) * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::sigmoid() {
  // Forward pass: compute sigmoid(x)
  double sigmoidData = 1.0 / (1.0 + std::exp(-this->data));
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(sigmoidData, std::move(prev), 's');

  // Backward pass: gradient of sigmoid(x)
  std::function<void()> add_backward = [this, newVal]() {
    // differentiation of sigmoid(x) => sigmoid(x) * (1-sigmoid(x))
    this->grad += newVal->data * (1.0 - newVal->data) * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::leakyRelu(double alpha) {
  // Forward pass: compute LeakyReLU(x)
  double leakyReluData = this->data > 0 ? this->data : alpha * this->data;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(leakyReluData, std::move(prev), 'l');

  // Backward pass: gradient of LeakyReLU(x)
  std::function<void()> add_backward = [this, alpha, newVal]() {
    double gradFactor = this->data > 0 ? 1.0 : alpha;
    this->grad += gradFactor * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::gelu() {
  // Constants for GELU approximation
  const double sqrt2OverPi = std::sqrt(2.0 / M_PI);
  const double coeff = 0.044715;

  // Forward pass: compute GELU(x)
  double tanhArg = sqrt2OverPi * (this->data + coeff * std::pow(this->data, 3));
  double geluData = 0.5 * this->data * (1.0 + std::tanh(tanhArg));
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(geluData, std::move(prev), 'g');

  // Backward pass: gradient of GELU(x) (approximation)
  std::function<void()> add_backward = [this, tanhArg, newVal, sqrt2OverPi]() {
    double tanhVal = std::tanh(tanhArg);
    double factor = 0.5 * (1.0 + tanhVal) +
        0.5 * this->data * (1.0 - tanhVal * tanhVal) * sqrt2OverPi *
            (1.0 + 3 * 0.044715 * this->data * this->data);
    this->grad += factor * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

// ========== functional non-linear operations
namespace NonLinear {

std::vector<std::shared_ptr<Value>> relu(
    std::vector<std::shared_ptr<Value>> inp) {
  std::vector<std::shared_ptr<Value>> out{};
  for (auto& e : inp) {
    std::shared_ptr<Value> curr = e->relu();
    out.push_back(curr);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> tanh(
    std::vector<std::shared_ptr<Value>> inp) {
  std::vector<std::shared_ptr<Value>> out{};
  for (auto& e : inp) {
    std::shared_ptr<Value> curr = e->tanh();
    out.push_back(curr);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> gelu(
    std::vector<std::shared_ptr<Value>> inp) {
  std::vector<std::shared_ptr<Value>> out{};
  for (auto& e : inp) {
    std::shared_ptr<Value> curr = e->gelu();
    out.push_back(curr);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> sigmoid(
    std::vector<std::shared_ptr<Value>> inp) {
  std::vector<std::shared_ptr<Value>> out{};
  for (auto& e : inp) {
    std::shared_ptr<Value> curr = e->sigmoid();
    out.push_back(curr);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> leakyRelu(
    double alpha,
    std::vector<std::shared_ptr<Value>> inp) {
  std::vector<std::shared_ptr<Value>> out{};
  for (auto& e : inp) {
    std::shared_ptr<Value> curr = e->leakyRelu(alpha);
    out.push_back(curr);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> softmax(
    std::vector<std::shared_ptr<Value>> inp) {
  // Step 1: Find the maximum value for numerical stability
  auto max_val = *std::max_element(
      inp.begin(),
      inp.end(),
      [](const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
        return a->data < b->data;
      });

  // Step 2: Compute exp(x_i - max_val) for each input
  std::vector<std::shared_ptr<Value>> exp_vals(inp.size());
  for (auto& val : inp) {
    exp_vals.push_back((val->sub(max_val))->exp());
  }

  // Step 3: Compute the sum of exp(x_i - max_val)
  std::shared_ptr<Value> sum_exp = std::make_shared<Value>(0.0);
  for (auto& exp_val : exp_vals) {
    sum_exp = sum_exp->add(exp_val);
  }

  // Step 4: Compute softmax = exp(x_i - max_val) / sum_exp
  std::vector<std::shared_ptr<Value>> softmax_vals(inp.size());
  for (auto& exp_val : exp_vals) {
    softmax_vals.push_back(exp_val->div(sum_exp));
  }

  return softmax_vals;
}

} // namespace NonLinear
