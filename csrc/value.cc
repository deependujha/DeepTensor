#include "value.h"
#include <cmath>
#include <unordered_set>
#include <vector>

/// BuildTopo
/// if not already visited the node, mark it visited, and then subsequently
/// traverse it's child nodes
void Value::build_topo(
    Value* v,
    std::unordered_set<Value*>& visited,
    std::vector<Value*>& topo_list) {
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
  std::vector<Value*> topo_list = {};
  std::unordered_set<Value*> visited;

  build_topo(this, visited, topo_list);

  // go one variable at a time and apply the chain rule to get its gradient
  this->grad = 1.0;

  // Iterating the vector in reverse order
  for (int i = int(topo_list.size()) - 1; i >= 0; i--) {
    topo_list[i]->executeBackWardMethod();
  }
}

Value* Value::add(Value* other) {
  double newData = this->data + other->data;
  std::unordered_set<Value*> prev = {this, other};
  Value* newVal = new Value(newData, prev, '+');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += newVal->grad;
    other->grad += newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

Value* Value::add(int other) {
  double newData = this->data + other;
  std::unordered_set<Value*> prev = {this};
  Value* newVal = new Value(newData, prev, '+');

  // Define the backward function
  std::function<void()> add_backward = [this, newVal]() {
    this->grad += newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

Value* Value::sub(Value* other) {
  double newData = this->data - other->data;
  std::unordered_set<Value*> prev = {this, other};
  Value* newVal = new Value(newData, prev, '-');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += newVal->grad;
    other->grad -= newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

Value* Value::sub(int other) {
  double newData = this->data - other;
  std::unordered_set<Value*> prev = {this};
  Value* newVal = new Value(newData, prev, '-');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

Value* Value::mul(Value* other) {
  double newData = this->data * other->data;
  std::unordered_set<Value*> prev = {this, other};
  Value* newVal = new Value(newData, prev, '*');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += other->data * newVal->grad;
    other->grad += this->data * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

Value* Value::mul(int other) {
  double newData = this->data * other;
  std::unordered_set<Value*> prev = {this};
  Value* newVal = new Value(newData, prev, '*');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += other * newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

Value* Value::pow(int n) {
  double newData = std::pow(this->data, n);
  std::unordered_set<Value*> prev = {this};
  Value* newVal = new Value(newData, prev, 'e');

  // Define the backward function
  std::function<void()> add_backward = [this, n, newVal]() {
    this->grad +=
        (n * std::pow(this->data, n - 1)) * newVal->grad; // n * (x^(n-1))
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}

Value* Value::relu() {
  double newData = this->data < 0 ? 0 : this->data;
  std::unordered_set<Value*> prev = {this};
  Value* newVal = new Value(newData, prev, 'r');

  // Define the backward function
  std::function<void()> add_backward = [this, newVal]() {
    this->grad += newVal->grad * (newVal->data > 0);
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}
