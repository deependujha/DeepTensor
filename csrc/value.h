#pragma once
#include <functional>
#include <unordered_set>

class Value {
private:
  //   internal properties
  std::unordered_set<Value*> _prev;
  char _op; // the op that produced this node
  // Function pointer for the private method

  // function type - returns void, and takes no parameter `()`
  std::function<void()> backward_;

public:
  double data;
  double grad = 0.0;
  Value(double data, std::unordered_set<Value*> _prev = {}, char _op = '-')
      : data(data), _prev(std::move(_prev)), _op(_op) {}

  // Setter to assign a new function
  void setBackWardMethod(std::function<void()> func) {
    backward_ = func;
  }

  // Method to execute the private method
  void executeBackWardMethod() {
    if (backward_) {
      backward_();
    }
  }

  Value* add(Value* other);
};
