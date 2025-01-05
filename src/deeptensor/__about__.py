# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import time

__version__ = "0.1.0"
__author__ = "Deependu Jha"
__author_email__ = "deependujha21@gmail.com"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2023-{time.strftime('%Y')}, {__author__}."
__homepage__ = "https://github.com/deependujha/deeptensor"
__docs_url__ = "https://deependujha.github.io/deeptensor"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = (
    "The Deep Learning framework to train, deploy, and ship AI products blazing fast."
)
__long_doc__ = """
DeepTensor: A Minimal Autograd Engine for Neural Networks
=========================================================

DeepTensor is a lightweight, minimalist autograd engine designed for implementing and training neural networks. The project is inspired by PyTorch's autograd system but aims to provide a simple, comprehensible implementation. It serves as a learning tool and a foundation for those who want to dive deep into the mechanics of automatic differentiation.

Key Features
------------
- **Dynamic Computation Graph**: DeepTensor builds computation graphs dynamically, allowing flexibility in defining and modifying models on the fly.
- **Custom Gradient Functions**: Define custom operations with gradients to suit your specific needs.
- **Python Bindings**: Built in C++ for performance, but fully accessible from Python using `pybind11`.
- **Extensible**: Modular design allows you to extend the library to support custom operations and data types.
- **Educational Focus**: Ideal for developers and researchers who want to understand the internals of an autograd engine.

Why DeepTensor?
---------------
DeepTensor is perfect for:
- Developers curious about the inner workings of autograd and neural network training.
- Educators seeking a simplified tool to demonstrate automatic differentiation.
- Experimenting with new ideas for differentiation and optimization algorithms.

Example Usage
-------------
Define a custom computation and compute gradients:

```python
import deeptensor as dt

# Define tensors
a = dt.Value(2.0)
b = dt.Value(3.0)

# Perform operations
c = a * b + b ** 2

# Backpropagate gradients
c.backward()

# Inspect gradients
print(a.grad)  # Gradient of c with respect to a
print(b.grad)  # Gradient of c with respect to b
```

Installation
------------
Install DeepTensor from PyPI:

```bash
pip install deeptensor
```

Getting Started
---------------
1. **Documentation**: Visit the [DeepTensor Documentation](https://github.com/deependujha/deeptensor#readme) for detailed guides and API references.
2. **Examples**: Check out the examples folder in the GitHub repository for practical use cases.

Contributing
------------
Contributions are welcome! If you want to add features, report bugs, or improve documentation, please visit the [GitHub repository](https://github.com/deependujha/deeptensor).

License
-------
DeepTensor is released under the MIT License.

"""


__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__docs_url__",
    "__homepage__",
    "__license__",
    "__version__",
]
