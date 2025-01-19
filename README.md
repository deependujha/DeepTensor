# DeepTensor

<div align="center">

<a href="https://pypi.org/project/deeptensor/"><img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white"/></a> <a href="https://deependujha.github.io/DeepTensor/"><img src="https://img.shields.io/badge/mkdocs-documentation"/></a>
![PyPI](https://img.shields.io/pypi/v/deeptensor)
![Downloads](https://img.shields.io/pypi/dm/deeptensor)
![License](https://img.shields.io/github/license/deependujha/DeepTensor)
<a target="_blank" href="https://colab.research.google.com/github/deependujha/DeepTensor/blob/main/demo/roboflow-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</div>

![mexican cat dance](https://www.deependujha.xyz/deeptensor-assets/mexican-cat-dance.gif)

- DeepTensor: A minimal deep learning library focused on custom autograd and efficient tensor operations.

---

## Installation

```bash
pip install deeptensor
```

---

## Checkout Demo

- [play with demo](./demo/main.ipynb)

![demo](https://www.deependujha.xyz/deeptensor-assets/deeptensor-confusion-matrix.png)

---

## Check Docs

- [visit docs](https://deependujha.github.io/deeptensor)

---

## Basic Usage

```python
from deeptensor import (
    # model
    Model,

    # Layers
    FeedForwardLayer,

    # activation layers
    GeLu,
    LeakyReLu,
    ReLu,
    Sigmoid,
    SoftMax,
    Tanh,

    # core objects
    Tensor,
    Value,

    # optimizers
    SGD,
    Momentum,
    AdaGrad,
    RMSprop,
    Adam,

    # losses
    mean_squared_error,
    cross_entropy,
    binary_cross_entropy,
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
)

opt = Adam(model, 0.01) # learning rate

print(model)

tensor_input = Tensor([2])
tensor_input.set(0, Value(2.4))
tensor_input.set(1, Value(5.2))

out = model(tensor_input)

loss = mean_squared_error(out, YOUR_EXPECTED_OUTPUT)

# backprop
loss.backward()
opt.step()
opt.zero_grad()
```

---

## WIP

- Save & Load model
- Train MNIST model
- Train a character-level transformer model
- Add support for DDP
- Add support for CUDA execution
