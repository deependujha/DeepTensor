#include "value.h"
#include <cmath>
#include <iostream>
#include <memory>
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
  for (int i = int(topo_list.size()) - 1; i >= 0; i--) {
    topo_list[i]->executeBackWardMethod();
  }
}

std::shared_ptr<Value> Value::add(std::shared_ptr<Value> other) {
  double newData = this->data + other->data;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this(), other};
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, '+');

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
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, '+');

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
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, '-');

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
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, '-');

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
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, '*');

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
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, '*');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += other * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::pow(int n) {
  double newData = std::pow(this->data, n);
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, 'e');

  // Define the backward function
  std::function<void()> add_backward = [this, n, newVal]() {
    this->grad +=
        (n * std::pow(this->data, n - 1)) * newVal->grad; // n * (x^(n-1))
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
    this->grad += newVal->grad * (newVal->data > 0);
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

std::shared_ptr<Value> Value::neg() {
  double newData = this->data * -1;
  std::unordered_set<std::shared_ptr<Value>> prev = {shared_from_this()};
  std::shared_ptr<Value> newVal = std::make_shared<Value>(newData, prev, 'n');

  return newVal;
}
