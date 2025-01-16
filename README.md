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
