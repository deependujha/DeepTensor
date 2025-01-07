#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "layers/feed_forward_layer.h"
#include "neural_network.h"
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
              std::shared_ptr<Value>)>(&Value::mul),
          "subtract value object with value object")
      .def(
          "__rsub__",
          static_cast<std::shared_ptr<Value> (Value::*)(
              std::shared_ptr<Value>)>(&Value::mul),
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

  //   exposing Layer class
  py::class_<FeedForwardLayer, std::shared_ptr<FeedForwardLayer>>(m, "FeedForwardLayer")
      .def(py::init<int, int>())
      .def(py::init<int, int, int>())
      .def("zero_grad", &FeedForwardLayer::zero_grad)
      .def("__call__", &FeedForwardLayer::call)
      .def("__repr__", &FeedForwardLayer::printMe);

  //   exposing MLP class
  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init<std::vector<std::shared_ptr<Layer>>, bool>())
      .def("zero_grad", &Model::zero_grad)
      .def("save_model", &Model::save_model)
      .def("load_model", &Model::load_model)
      .def("__call__", &Model::call)
      .def("__repr__", &Model::printMe);
}
