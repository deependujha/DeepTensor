#include "value.h"
#include <unordered_set>

Value* Value::add(Value* other) {
  double newData = this->data + other->data;
  std::unordered_set<Value*> prev = {this, other};
  Value* newVal = new Value(this->data + other->data, prev, '+');

  // Define the backward function
  std::function<void()> add_backward = [this, other, newVal]() {
    this->grad += newVal->grad;
    other->grad += newVal->grad;
  };

  newVal->setBackWardMethod(add_backward);

  return newVal;
}
