#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

  //   exposing Neuron class
  py::class_<Neuron, std::shared_ptr<Neuron>>(m, "Neuron")
      .def(py::init<int>())
      .def(py::init<int, bool>())
      .def(py::init<int, bool, int>())
      .def("parameters", &Neuron::parameters)
      .def("zero_grad", &Neuron::zero_grad)
      .def("__call__", &Neuron::call)
      .def("__repr__", &Neuron::printMe);

  //   exposing Layer class
  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def(py::init<int, int>())
      .def(py::init<int, int, bool>())
      .def(py::init<int, int, bool, int>())
      .def("parameters", &Layer::parameters)
      .def("zero_grad", &Layer::zero_grad)
      .def("__call__", &Layer::call)
      .def("__repr__", &Layer::printMe);

  //   exposing MLP class
  py::class_<MLP, std::shared_ptr<MLP>>(m, "MLP")
      .def(py::init<int, std::vector<int>>())
      .def(py::init<int, std::vector<int>, bool>())
      .def(py::init<int, std::vector<int>, bool, int>())
      .def("parameters", &MLP::parameters)
      .def("zero_grad", &MLP::zero_grad)
      .def("__call__", &MLP::call)
      .def("__repr__", &MLP::printMe);
}
