# DeepTensor 🔥

<div align="center">

<a href="https://pypi.org/project/deeptensor/"><img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white"/></a> <a href="https://deependujha.github.io/DeepTensor/"><img src="https://img.shields.io/badge/mkdocs-documentation"/></a>
![PyPI](https://img.shields.io/pypi/v/deeptensor)
![Downloads](https://img.shields.io/pypi/dm/deeptensor)
![License](https://img.shields.io/github/license/deependujha/DeepTensor)
<a target="_blank" href="https://colab.research.google.com/github/deependujha/DeepTensor/blob/main/demo/roboflow-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</div>

<div align="center">

![mexican cat dance](https://www.deependujha.xyz/deeptensor-assets/mexican-cat-dance.gif)

</div>

- **`DeepTensor`**: A minimal PyTorch-like **deep learning library** focused on custom autograd and efficient tensor operations.

---

## **Features at a Glance** 🚀

- **Automatic gradient computation** with a custom autograd engine.
- **Weight initialization schemes**:
  - `Xavier/Glorot` and `He` initialization in both `uniform` and `normal` variants.
- **Activation functions**:
  - `ReLU`, `GeLU`, `Sigmoid`, `Tanh`, `SoftMax`, `LeakyReLU`, and more.
- **Built-in loss functions**:
  - `Mean Squared Error (MSE)`, `Cross Entropy`, and `Binary Cross Entropy`.
- **Optimizers**:
  - `SGD`, `Momentum`, `AdaGrad`, `RMSprop`, and `Adam`.

---

### **Why DeepTensor?**

DeepTensor offers a hands-on implementation of deep learning fundamentals with a focus on **customizability** and **learning the internals** of deep learning frameworks like PyTorch.

---

## Installation

```bash
pip install deeptensor
```

---

## Setup the project for development

```bash
git clone --recurse-submodules -j8 git@github.com:deependujha/DeepTensor.git
cd DeepTensor

# run ctests
make ctest

# install python package in editable mode
pip install -e .

# run pytest
make test
```

---

## Checkout Demo

- [play with latest demo](./demo/roboflow-demo.ipynb)

![demo](https://www.deependujha.xyz/deeptensor-assets/deeptensor-confusion-matrix.png)

---

## Check Docs

- [visit docs](https://deependujha.github.io/DeepTensor)

![loss curve](https://www.deependujha.xyz/deeptensor-assets/loss-curve.png)

---

## Basic Usage

```python
from deeptensor import (
    # model
    Model,

    # Layers
    Conv2D,
    MaxPooling2D,
    Flatten,
    LinearLayer,

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
        LinearLayer(2, 16),
        ReLu(),
        LinearLayer(16, 16),
        LeakyReLu(0.1),
        LinearLayer(16, 1),
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

## Features expected to be added

- Save & Load model
- Train a character-level transformer model
- Add support for DDP
- Add support for CUDA execution ⭐️

---

## Open to Opportunities 🎅🏻🎁

I am actively seeking new opportunities to contribute to impactful projects in the deep learning and AI space.

If you are interested in collaborating or have a position that aligns with my expertise, feel free to reach out!

You can connect with me on [GitHub](https://github.com/deependujha), [X (formerly twitter)](https://x.com/deependu__), or email me: `deependujha21@gmail.com`.
