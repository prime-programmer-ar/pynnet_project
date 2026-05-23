<p align="center">
  <h1 align="center">🧠 PyNNet</h1>
  <p align="center"><strong>An Educational Deep Learning Library — Built from Scratch</strong></p>
  <p align="center">
    <a href="https://pypi.org/project/pynnet/"><img src="https://img.shields.io/pypi/v/pynnet?color=blue&label=PyPI" alt="PyPI"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python"></a>
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/status-beta-green" alt="Status"></a>
    <a href="#"><img src="https://img.shields.io/badge/tests-36%20passed-brightgreen" alt="Tests"></a>
  </p>
</p>

---

**PyNNet** is a beginner-friendly neural network library that helps you **learn and understand deep learning from the ground up**. Built entirely in Python and NumPy, every line of code is designed to be read, understood, and learned from.

> 💡 **Not a replacement for PyTorch or TensorFlow** — PyNNet is purpose-built for **education**. Read the source code. Understand backpropagation. Build intuition.

---

## ✨ What's New in v1.0.1

- 🏗️ **CNN Support** — `conv2d`, `maxpool2d`, and `flatten` layers
- 🔁 **RNN Support** — Full `lstm` layer with BPTT
- 🎛️ **Regularization** — `dropout` and `batchnorm` layers
- 🔧 **4 Optimizers** — SGD, Momentum, RMSprop, Adam
- 🛡️ **Numerical Stability** — Stable sigmoid, softplus, and loss functions
- 📊 **`model.summary()`** — View architecture and parameter counts
- ✅ **36 unit tests** — Full test coverage with pytest

---

## 📦 Installation

```bash
pip install pynnet
```

**Requirements:** Python 3.8+ and NumPy (installed automatically).

---

## 🚀 Quick Start

### Build an MLP (Multi-Layer Perceptron)

```python
import numpy as np
from pynnet.network import sequential
from pynnet.layers import dense
from pynnet.activation import activation, relu, relu_derivative, sigmoid, sigmoid_derivative
from pynnet.optimizer import Adam
from pynnet.loss import mse, mse_derivative

# XOR dataset
X = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]], dtype=float)
y = np.array([[[0]],   [[1]],   [[1]],   [[0]]], dtype=float)

# Build model
model = sequential()
model.add(dense(input_size=2, output_size=8, weight_init='he'))
model.add(activation(relu, relu_derivative))
model.add(dense(input_size=8, output_size=1, weight_init='xavier'))
model.add(activation(sigmoid, sigmoid_derivative))

# Compile & train
model.compile(loss=mse, loss_derivative=mse_derivative, optimizer=Adam(learning_rate=0.01))
model.fit(X, y, epochs=1000, verbose=True, print_every=200)

# Predict
print(model.predict(X))
```

### Build a CNN

```python
from pynnet.network import sequential
from pynnet.layers import conv2d, maxpool2d, flatten, dense
from pynnet.activation import activation, relu, relu_derivative, softmax
from pynnet.optimizer import Adam
from pynnet.loss import categorical_cross_entropy, cce_derivative_with_softmax

model = sequential()

# Convolutional block
model.add(conv2d(num_filters=8, kernel_size=3, input_channels=1))   # (1,28,28) → (8,26,26)
model.add(activation(relu, relu_derivative))
model.add(maxpool2d(pool_size=2))                                    # (8,26,26) → (8,13,13)

# Classifier
model.add(flatten())                                                 # (8,13,13) → (1,1352)
model.add(dense(input_size=1352, output_size=10, weight_init='he'))
model.add(softmax())

model.compile(
    loss=categorical_cross_entropy,
    loss_derivative=cce_derivative_with_softmax,
    optimizer=Adam(learning_rate=0.001)
)
```

### Build with LSTM

```python
from pynnet.layers import lstm, dense
from pynnet.optimizer import Adam

rnn = lstm(input_size=10, hidden_size=32, return_sequences=False)
fc = dense(input_size=32, output_size=5)

# Process a sequence: (seq_len=20, features=10) → (1, 5)
```

---

## 🛠️ Feature Reference

### Layers

| Layer | Description | Import |
|-------|-------------|--------|
| `dense` | Fully connected layer | `from pynnet.layers import dense` |
| `conv2d` | 2D convolution with padding & stride | `from pynnet.layers import conv2d` |
| `maxpool2d` | 2D max pooling | `from pynnet.layers import maxpool2d` |
| `flatten` | Reshape (C,H,W) → (1, C×H×W) | `from pynnet.layers import flatten` |
| `dropout` | Inverted dropout regularization | `from pynnet.layers import dropout` |
| `batchnorm` | Batch normalization | `from pynnet.layers import batchnorm` |
| `lstm` | LSTM recurrent layer | `from pynnet.layers import lstm` |

### Activation Functions

| Function | Best For | Derivative |
|----------|----------|------------|
| `relu` | Hidden layers (default) | `relu_derivative` |
| `sigmoid` | Binary output | `sigmoid_derivative` |
| `tanh` | Hidden layers (RNN) | `tanh_derivative` |
| `linear` | Regression output | `linear_derivative` |
| `leaky_relu` | Avoiding dead neurons | `leaky_relu_derivative` |
| `elu` | Smooth alternative to ReLU | `elu_derivative` |
| `swish` | Modern networks | `swish_derivative` |
| `softmax` | Multi-class output | *(special layer)* |

### Loss Functions

| Function | Use Case |
|----------|----------|
| `mse` / `mse_derivative` | Regression |
| `mae` / `mae_derivative` | Robust regression |
| `huber_loss` / `huber_loss_derivative` | Outlier-resistant regression |
| `binary_cross_entropy` / `binary_cross_entropy_derivative` | Binary classification |
| `categorical_cross_entropy` / `cce_derivative_with_softmax` | Multi-class classification |

### Optimizers

| Optimizer | Description |
|-----------|-------------|
| `SGD(lr)` | Vanilla gradient descent |
| `Momentum(lr, momentum)` | SGD with momentum |
| `RMSprop(lr, rho)` | Adaptive learning rates |
| `Adam(lr, beta1, beta2)` | Adaptive moments (recommended) |

### Weight Initialization

| Method | Best With |
|--------|-----------|
| `'he'` | ReLU / Leaky ReLU *(default)* |
| `'xavier'` | Sigmoid / Tanh |
| `'lecun'` | SELU / normalized inputs |
| `'orthogonal'` | Deep networks / RNNs |

---

## 📂 Project Structure

```
pynnet/
├── layers/
│   ├── base.py        # Abstract base layer
│   ├── dense.py       # Fully connected layer
│   ├── conv2d.py      # 2D convolution
│   ├── maxpool2d.py   # Max pooling
│   ├── flatten.py     # Flatten layer
│   ├── dropout.py     # Dropout regularization
│   ├── batchnorm.py   # Batch normalization
│   └── lstm.py        # LSTM recurrent layer
├── activation.py      # Activation functions & layers
├── loss.py            # Loss functions & derivatives
├── optimizer.py       # SGD, Momentum, RMSprop, Adam
├── network.py         # Sequential model class
└── __init__.py        # Public API exports
tests/
└── test_pynnet.py     # 36 unit tests
examples/
├── 01_xor_example.py
├── 02_binary_classification.py
└── 03_regression.py
```

---

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 💾 Save & Load Models

```python
# Save weights
model.save_weights('my_model.npz')

# Load into an identical architecture
model.load_weights('my_model.npz')
```

---

## 📋 Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0

---

## 📄 License

[Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — free for non-commercial and educational use.

## 👤 Author

**Zain Qamar** — [GitHub](https://github.com/prime-programmer-ar) · [Email](mailto:zainqamarch@gmail.com)

## 📖 Citation

```bibtex
@software{pynnet2025,
  author = {Qamar, Zain},
  title  = {PyNNet: An Educational Deep Learning Library},
  year   = {2025},
  url    = {https://github.com/prime-programmer-ar/pynnet_project}
}
```
