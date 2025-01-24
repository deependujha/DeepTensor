#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

class Value : public std::enable_shared_from_this<Value> {
private:
  //   internal properties
  // Function pointer for the private method

  // function type - returns void, and takes no parameter `()`
  std::function<void()> backward_ = nullptr;

  void build_topo(
      std::shared_ptr<Value> v,
      std::unordered_set<std::shared_ptr<Value>>& visited,
      std::vector<std::shared_ptr<Value>>& topo_list);

public:
  double data = 0.0;
  double grad = 0.0;
  std::unordered_set<std::shared_ptr<Value>> _prev = {};
  char _op = '-'; // the op that produced this node

  Value(double data) : data(data) {}
  Value(double data, std::unordered_set<std::shared_ptr<Value>> _prev, char _op)
      : data(data), _prev(std::move(_prev)), _op(_op) {}
  
  ~Value(){
    this->_prev.clear();
    this->clearBackwardMethod();
  }

  // Setter to assign a new function
  void setBackWardMethod(std::function<void()> func) {
    this->backward_ = func;
  }

  // Method to execute the private method backward_
  void executeBackWardMethod() {
    if (this->backward_) {
      this->backward_();
    }
  }

  void clearBackwardMethod() {
    this->backward_ = nullptr;
  }

  void backward();

  std::string printMe() {
    std::string s = "Value(data=" + std::to_string(this->data) +
        ", grad=" + std::to_string(this->grad) + ")";
    return s;
  }

  std::shared_ptr<Value> add(std::shared_ptr<Value> other);
  std::shared_ptr<Value> add(double other);

  std::shared_ptr<Value> sub(std::shared_ptr<Value> other);
  std::shared_ptr<Value> sub(double other);

  std::shared_ptr<Value> mul(std::shared_ptr<Value> other);
  std::shared_ptr<Value> mul(double other);
  std::shared_ptr<Value> div(std::shared_ptr<Value> other);
  std::shared_ptr<Value> div(double other);
  std::shared_ptr<Value> rdiv(double other);

  std::shared_ptr<Value> pow(int n);
  std::shared_ptr<Value> neg();
  std::shared_ptr<Value> exp();
  std::shared_ptr<Value> ln();

  // non-linear functions
  std::shared_ptr<Value> relu();
  std::shared_ptr<Value> tanh();
  std::shared_ptr<Value> gelu();
  std::shared_ptr<Value> sigmoid();
  std::shared_ptr<Value> leakyRelu(double alpha);
};
