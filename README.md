# DeepTensor

![mexican cat dance](./assets/mexican-cat-dance.gif)

- DeepTensor: A minimal deep learning library focused on custom autograd and efficient tensor operations.

---

## Installation

```bash
pip install deeptensor
```

---

## Checkout Demo

- [play with demo](./demo/main.ipynb)

![demo](./assets/trained-model.png)

---

## Check Docs

- [visit docs](https://deependujha.github.io/deeptensor)

---

## Basic Usage

```python
from deeptensor import (
    FeedForwardLayer,
    GeLu,
    LeakyReLu,
    Model,
    ReLu,
    Sigmoid,
    SoftMax,
    Tanh,
    Tensor,
    Value,
)

model = Model(
    [
        FeedForwardLayer(2, 16),
        ReLu(),
        FeedForwardLayer(16, 16),
        LeakyReLu(0.1),
        FeedForwardLayer(16, 1),
        Sigmoid(),
    ],
    False,  # using_cuda
)  # 2-layer neural network

print(model)

tensor_input = Tensor([2])
tensor_input.set(0, Value(2.4))
tensor_input.set(1, Value(5.2))

out = model(tensor_input)
```

---

## WIP

- Save & Load model
- Train MNIST model
- Train a character-level transformer model
- Add support for DDP
- Add support for CUDA execution

---

## To be done later

- Add support for optimizers (sgd, nesterov, adam, etc)
