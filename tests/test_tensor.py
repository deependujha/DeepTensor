from __future__ import annotations

from math import isclose

from deeptensor import Tensor, Value


def test_tensor_set_and_get_one_d():
    # ------------ 1D tensor ------------
    t1 = Tensor([4])
    vals = [1.0, 2.0, 3.0, 4.0]

    for i, val in enumerate(vals):
        t1.set(i, Value(val))

    for i in range(len(vals)):
        assert isclose(t1.get(i).data, vals[i])


def test_tensor_set_and_get_two_d():
    # ------------ 2D tensor ------------
    t1 = Tensor([2, 3])
    vals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    counter = 1.0
    for i in range(2):
        for j in range(3):
            t1.set([i, j], Value(counter))
            counter += 1.0

    for i in range(2):
        for j in range(3):
            assert t1.get([i, j]).data == vals[i][j]


def test_add_tensor():
    t1 = Tensor([3])
    t2 = Tensor([3])

    t1.set(0, Value(1.0))
    t1.set(1, Value(2.0))
    t1.set(2, Value(3.0))

    t2.set(0, Value(10.0))
    t2.set(1, Value(20.0))
    t2.set(2, Value(30.0))

    t3 = t1 + t2

    assert isclose(t3.get(0).data, 11.0)
    assert isclose(t3.get(1).data, 22.0)
    assert isclose(t3.get(2).data, 33.0)


def test_div_tensor():
    t1 = Tensor([3])

    t1.set(0, Value(10.0))
    t1.set(1, Value(20.0))
    t1.set(2, Value(30.0))

    t2 = t1 / Value(5)

    assert isclose(t2.get(0).data, 2.0)
    assert isclose(t2.get(1).data, 4.0)
    assert isclose(t2.get(2).data, 6.0)


def test_matmul():
    # Create two matrices as Tensors
    t1 = Tensor([2, 2])
    t2 = Tensor([2, 2])
    expected = Tensor([2, 2])

    val1 = [[1.0, 2.0], [3.0, 4.0]]
    val2 = [[5.0, 6.0], [7.0, 8.0]]

    # Expected result of t1 @ t2
    # [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    # [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    expected_val = [[19.0, 22.0], [43.0, 50.0]]

    for i in range(2):
        for j in range(2):
            t1.set([i, j], Value(val1[i][j]))
            t2.set([i, j], Value(val2[i][j]))
            expected.set([i, j], Value(expected_val[i][j]))

    # Perform matrix multiplication
    t3 = t1.matmul(t2)  # or t3 = t1 * t2 if overloaded

    # Check if the result matches the expected tensor
    for i in range(2):
        for j in range(2):
            assert (
                t3.get([i, j]).data == expected_val[i][j]
            ), f"Matrix multiplication failed: {t3} != {expected}"
