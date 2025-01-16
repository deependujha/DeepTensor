#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "layers/feed_forward_layer.h"
#include "layers/non_linear_layer.h"
#include "neural_network.h"
#include "optimizer.h"
#include "tensor.h"
#include "value.h"

namespace py = pybind11;
using overload_cast_ = pybind11::detail::overload_cast_impl<Value>;

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "A minimal deep learning framework made by Deependu Jha <deependujha21@gmail.com>"; // optional module docstring
  py::class_<Value, std::shared_ptr<Value>>(m, "Value")
      .def(py::init<double>())
      .def(py::init<double, std::unordered_set<std::shared_ptr<Value>>, char>())
      .def_readwrite("data", &Value::data)
      .def_readwrite("grad", &Value::grad)
      .def_readwrite("_prev", &Value::_prev)
      .def_readwrite("char", &Value::_op)
      .def("backward", &Value::backward)
      .def("executeBackward", &Value::executeBackWardMethod)
      .def("__repr__", &Value::printMe)
      .def(
          "__add__",
          static_cast<std::shared_ptr<Value> (Value::*)(double)>(&Value::add),
          "add value object with double")
      .def(
          "__radd__",
          static_cast<std::shared_ptr<Value> (Value::*)(double)>(&Value::add),
          "add value object with double")
      .def(
          "__add__",
          static_cast<std::shared_ptr<Value> (Value::*)(
              std::shared_ptr<Value>)>(&Value::add),
          "add value object with value object")
      .def(
          "__radd__",
          static_cast<std::shared_ptr<Value> (Value::*)(
              std::shared_ptr<Value>)>(&Value::add),
          "add value object with value object")
      .def(
          "__sub__",
          static_cast<std::shared_ptr<Value> (Value::*)(double)>(&Value::sub),
          "subtract value object with double")
      .def(
          "__rsub__",
          static_cast<std::shared_ptr<Value> (Value::*)(double)>(&Value::sub),
          "subtract value object with double")
      .def(
          "__sub__",
          static_cast<std::shared_ptr<Value> (Value::*)(
              std::shared_ptr<Value>)>(&Value::sub),
          "subtract value object with value object")
      .def(
          "__rsub__",
          static_cast<std::shared_ptr<Value> (Value::*)(
              std::shared_ptr<Value>)>(&Value::sub),
          "subtract value object with value object")
      .def(
          "__mul__",
          static_cast<std::shared_ptr<Value> (Value::*)(double)>(&Value::mul),
          "multiply value object with double")
      .def(
          "__rmul__",
          static_cast<std::shared_ptr<Value> (Value::*)(double)>(&Value::mul),
          "multiply value object with double")
      .def(
          "__mul__",
          static_cast<std::shared_ptr<Value> (Value::*)(
              std::shared_ptr<Value>)>(&Value::mul),
          "multiply value object with value object")
      .def(
          "__rmul__",
          static_cast<std::shared_ptr<Value> (Value::*)(
              std::shared_ptr<Value>)>(&Value::mul),
          "multiply value object with value object")
      .def(
          "__pow__",
          static_cast<std::shared_ptr<Value> (Value::*)(int)>(&Value::pow),
          "raise power of value object by int n")
      .def(
          "__neg__",
          static_cast<std::shared_ptr<Value> (Value::*)()>(&Value::neg),
          "negative of the value object")
      .def(
          "relu",
          static_cast<std::shared_ptr<Value> (Value::*)()>(&Value::relu),
          "apply relu operation");

  // exposing tensor class
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init<std::vector<int>>())
      .def(
          "set",
          static_cast<void (Tensor::*)(
              std::vector<int>, std::shared_ptr<Value>)>(&Tensor::set))
      .def(
          "set",
          static_cast<void (Tensor::*)(int, std::shared_ptr<Value>)>(
              &Tensor::set))
      .def(
          "get",
          static_cast<std::shared_ptr<Value> (Tensor::*)(int)>(&Tensor::get))
      .def(
          "get",
          static_cast<std::shared_ptr<Value> (Tensor::*)(std::vector<int>)>(
              &Tensor::get))
      .def("normalize_idx", &Tensor::normalize_idx)
      .def("backward", &Tensor::backward)
      .def("zero_grad", &Tensor::zero_grad)
      .def("add", &Tensor::add)
      .def("div", &Tensor::div)
      .def("matmul", &Tensor::matmul)
      .def("relu", &Tensor::relu)
      .def("gelu", &Tensor::gelu)
      .def("sigmoid", &Tensor::sigmoid)
      .def("tanh", &Tensor::tanh)
      .def("leakyRelu", &Tensor::leakyRelu)
      .def("softmax", &Tensor::softmax)
      .def("__repr__", &Tensor::printMe);

  //   exposing Layer class
  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def("zero_grad", &Layer::zero_grad)
      .def("__call__", &Layer::call)
      .def("parameters", &Layer::parameters)
      .def("__repr__", &Layer::printMe);

  py::class_<FeedForwardLayer, Layer, std::shared_ptr<FeedForwardLayer>>(
      m, "FeedForwardLayer")
      .def(py::init<int, int>())
      .def(py::init<int, int, int>())
      .def(py::init<int, int, int, std::string, std::string>())
      .def("zero_grad", &FeedForwardLayer::zero_grad)
      .def("parameters", &FeedForwardLayer::parameters)
      .def("__call__", &FeedForwardLayer::call)
      .def("__repr__", &FeedForwardLayer::printMe);

  py::class_<ReLu, Layer, std::shared_ptr<ReLu>>(m, "ReLu")
      .def(py::init<>())
      .def("zero_grad", &ReLu::zero_grad)
      .def("__call__", &ReLu::call)
      .def("parameters", &ReLu::parameters)
      .def("__repr__", &ReLu::printMe);

  py::class_<GeLu, Layer, std::shared_ptr<GeLu>>(m, "GeLu")
      .def(py::init<>())
      .def("zero_grad", &GeLu::zero_grad)
      .def("__call__", &GeLu::call)
      .def("parameters", &GeLu::parameters)
      .def("__repr__", &GeLu::printMe);

  py::class_<Sigmoid, Layer, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
      .def(py::init<>())
      .def("zero_grad", &Sigmoid::zero_grad)
      .def("__call__", &Sigmoid::call)
      .def("parameters", &Sigmoid::parameters)
      .def("__repr__", &Sigmoid::printMe);

  py::class_<Tanh, Layer, std::shared_ptr<Tanh>>(m, "Tanh")
      .def(py::init<>())
      .def("zero_grad", &Tanh::zero_grad)
      .def("__call__", &Tanh::call)
      .def("parameters", &Tanh::parameters)
      .def("__repr__", &Tanh::printMe);

  py::class_<LeakyReLu, Layer, std::shared_ptr<LeakyReLu>>(m, "LeakyReLu")
      .def(py::init<double>())
      .def("zero_grad", &LeakyReLu::zero_grad)
      .def("__call__", &LeakyReLu::call)
      .def("parameters", &LeakyReLu::parameters)
      .def("__repr__", &LeakyReLu::printMe);

  py::class_<SoftMax, Layer, std::shared_ptr<SoftMax>>(m, "SoftMax")
      .def(py::init<>())
      .def("zero_grad", &SoftMax::zero_grad)
      .def("__call__", &SoftMax::call)
      .def("parameters", &SoftMax::parameters)
      .def("__repr__", &SoftMax::printMe);

  //   exposing Model class
  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init<std::vector<std::shared_ptr<Layer>>, bool>())
      .def_readwrite("using_cuda", &Model::using_cuda)
      .def_readwrite("layers", &Model::layers)
      .def("zero_grad", &Model::zero_grad)
      .def("save_model", &Model::save_model)
      .def("load_model", &Model::load_model)
      .def("parameters", &Model::parameters)
      .def("__call__", &Model::call)
      .def("__repr__", &Model::printMe);

  //   Optimzer class
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
      .def("step", &Optimizer::step);

  py::class_<SGD, std::shared_ptr<SGD>>(m, "SGD")
      .def(py::init<std::shared_ptr<Model>, double>())
      .def_readwrite("learning_rate", &SGD::learning_rate)
      .def("step", &SGD::step);

  py::class_<Momentum, std::shared_ptr<Momentum>>(m, "Momentum")
      .def(py::init<std::shared_ptr<Model>, double, double>())
      .def_readwrite("learning_rate", &Momentum::learning_rate)
      .def_readwrite("decay_factor", &Momentum::decay_factor)
      .def("step", &Momentum::step);

  py::class_<AdaGrad, std::shared_ptr<AdaGrad>>(m, "AdaGrad")
      .def(py::init<std::shared_ptr<Model>, double>())
      .def_readwrite("learning_rate", &AdaGrad::learning_rate)
      .def("step", &AdaGrad::step);

  py::class_<RMSprop, std::shared_ptr<RMSprop>>(m, "RMSprop")
      .def(py::init<std::shared_ptr<Model>, double>())
      .def(py::init<std::shared_ptr<Model>, double, double>())
      .def_readwrite("learning_rate", &RMSprop::learning_rate)
      .def_readwrite("decay_factor", &RMSprop::decay_factor)
      .def("step", &RMSprop::step);

  py::class_<Adam, std::shared_ptr<Adam>>(m, "Adam")
      .def(py::init<std::shared_ptr<Model>, double>())
      .def(py::init<std::shared_ptr<Model>, double, double, double>())
      .def_readwrite("learning_rate", &Adam::learning_rate)
      .def_readwrite("beta1", &Adam::beta1)
      .def_readwrite("beta2", &Adam::beta2)
      .def("step", &Adam::step);
}
