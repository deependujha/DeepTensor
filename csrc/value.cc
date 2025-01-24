#include "value.h"
#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

/// BuildTopo
/// if not already visited the node, mark it visited, and then subsequently
/// traverse it's child nodes
void Value::build_topo(
    std::shared_ptr<Value> v,
    std::unordered_set<std::shared_ptr<Value>>& visited,
    std::vector<std::shared_ptr<Value>>& topo_list) {
  if (v == nullptr) {
    return;
  }
  if (visited.find(v) != visited.end()) {
    return;
  }

  visited.insert(v);

  for (auto& child : v->_prev) {
    if (visited.find(child) == visited.end()) {
      build_topo(child, visited, topo_list);
    }
  }

  topo_list.push_back(v);
}

void Value::backward() {
  std::vector<std::shared_ptr<Value>> topo_list = {};
  std::unordered_set<std::shared_ptr<Value>> visited;

  build_topo(shared_from_this(), visited, topo_list);

  // go one variable at a time and apply the chain rule to get its gradient
  this->grad = 1.0;

  // Iterating the vector in reverse order
  // std::cout << "topo list: \n";
  for (int i = int(topo_list.size()) - 1; i >= 0; i--) {
    // std::cout << "i: " << i << "; node: " << topo_list[i]->printMe() << "\n";
    topo_list[i]->executeBackWardMethod();
    topo_list[i]->clearBackwardMethod();
    topo_list[i]->_prev.clear();
  }
}

std::shared_ptr<Value> Value::add(std::shared_ptr<Value> other) {
  double newData = this->data + other->data;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this(), other};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '+');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += newVal->grad;
    other->grad += newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::add(double other) {
  double newData = this->data + other;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '+');

  // Define the backward function
  std::function<void()> add_backward = [this, newVal]() {
    this->grad += newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::sub(std::shared_ptr<Value> other) {
  double newData = this->data - other->data;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this(), other};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '-');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += newVal->grad;
    other->grad -= newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::sub(double other) {
  double newData = this->data - other;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '-');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::mul(std::shared_ptr<Value> other) {
  double newData = this->data * other->data;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this(), other};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '*');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += other->data * newVal->grad;
    other->grad += this->data * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::mul(double other) {
  double newData = this->data * other;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '*');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += other * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::div(std::shared_ptr<Value> other) {
  // Forward pass: compute the division
  assert(other->data != 0 && "Division by zero is not allowed.");

  double newData = this->data / other->data;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this(), other};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '/');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    // Backward pass: gradient of (x / other) is (1 / other)
    this->grad += (1.0 / other->data) * newVal->grad;
    // Backward pass: gradient of (other / x) is (-other / x^2)
    other->grad += (-this->data / (other->data * other->data)) * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::div(double other) {
  // Forward pass: compute the division
  assert(other != 0 && "Division by zero is not allowed.");

  double newData = this->data / other;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '/');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    // Backward pass: gradient of (x / other) is (1 / other)
    this->grad += (1.0 / other) * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::rdiv(double other) {
  // Forward pass: compute the division
  assert(this->data != 0 && "Division by zero is not allowed.");

  double newData = double(other) / double(this->data);
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), '/');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    // Backward pass: gradient of (other / x) is (- other/x^2)
    this->grad += (-other / (this->data * this->data)) * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::pow(int n) {
  double newData = std::pow(this->data, n);
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), 'e');

  // Define the backward function
  std::function<void()> add_backward = [this, n, newVal]() {
    this->grad +=
        (n * std::pow(this->data, n - 1)) * newVal->grad; // n * (x^(n-1))
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::neg() {
  double newData = this->data * -1;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), 'n');

  return newVal;
}

std::shared_ptr<Value> Value::exp() {
  double newData = std::exp(this->data);
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), 'e');

  // Define the backward function
  std::function<void()> add_backward = [this, newVal]() {
    this->grad += (newVal->data * newVal->grad); // e^x => e^x
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::ln() {
  if (this->data <= 0) {
    throw std::runtime_error(
        "Natural log is not defined for numbers less than or equal to 0. Got" +
        std::to_string(this->data));
  }
  double newData = std::log(this->data);
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal =
      std::make_shared<Value>(newData, std::move(prev), 'l');

  // Define the backward function
  std::function<void()> add_backward = [this, newVal]() {
    this->grad += ((1 / (this->data)) * newVal->grad); // ln(x) => 1/x
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

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
  double sqrt2OverPi = std::sqrt(2.0 / M_PI);
  double coeff = 0.044715;

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
