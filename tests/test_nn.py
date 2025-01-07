from __future__ import annotations

from math import isclose

from deeptensor import Value


def test_value_backward():
    v1 = Value(3.0)
    v2 = Value(4.0)
    result = v1 * v2
    result.backward()

    assert isclose(v1.grad, 4.0)
    assert isclose(v2.grad, 3.0)
