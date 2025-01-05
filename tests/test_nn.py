from __future__ import annotations

from math import isclose

from deeptensor import MLP, Layer, Neuron, Value


def test_value_backward():
    v1 = Value(3.0)
    v2 = Value(4.0)
    result = v1 * v2
    result.backward()

    assert isclose(v1.grad, 4.0)
    assert isclose(v2.grad, 3.0)


# Test for the Neuron class
def test_neuron_initialization():
    n = Neuron(3, True, 42)

    params = n.parameters()
    assert len(params) == 4  # 3 weights + 1 bias
    assert all(isinstance(p, Value) for p in params)


def test_neuron_forward_pass():
    n = Neuron(2, True, 42)

    v1 = Value(1.0)
    v2 = Value(2.0)
    result = n.call([v1, v2])

    assert isinstance(result, Value)
    assert result.data is not None


# Test for the Layer class
def test_layer_initialization():
    lr = Layer(3, 2, True, 42)

    params = lr.parameters()
    assert len(params) == 8  # (3 weights + 1 bias) * 2 neurons


def test_layer_forward_pass():
    lr = Layer(2, 3, True, 42)

    v1 = Value(1.0)
    v2 = Value(2.0)
    outputs = lr.call([v1, v2])

    assert len(outputs) == 3
    assert all(isinstance(o, Value) for o in outputs)


# Test for the MLP class
def test_mlp_initialization():
    mlp = MLP(3, [4, 2], True, 42)

    params = mlp.parameters()
    expected_param_count = (3 * 4 + 4) + (4 * 2 + 2)  # weights and biases
    assert len(params) == expected_param_count


def test_mlp_forward_pass():
    mlp = MLP(2, [3, 2], True, 42)

    v1 = Value(1.0)
    v2 = Value(2.0)
    outputs = mlp.call([v1, v2])

    assert len(outputs) == 2
    assert all(isinstance(o, Value) for o in outputs)
