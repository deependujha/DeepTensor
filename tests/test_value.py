from __future__ import annotations

from math import isclose

from deeptensor import Value


def test_value_usage():
    a = Value(5.2)
    b = Value(4.3)
    c = a + b
    assert isclose(c.data, a.data + b.data)

    assert a.grad == 0
    assert b.grad == 0
    assert c.grad == 0


def test_value_advanced_usage():
    a = Value(5.0)
    b = Value(4.0)

    c = a**2 + b * 5 - 1

    assert isclose(c.data, 44)  # 5**2 + 4*5 -1 == 44

    c.backward()

    assert isclose(c.grad, 1)
    assert isclose(a.grad, 10)  # 2*a*c_grad
    assert isclose(b.grad, 5)  # 5*c_grad
