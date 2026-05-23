"""
pynnet — An Educational Deep Learning Library
===============================================

A beginner-friendly neural network library built from scratch in
Python and NumPy. Provides clean, transparent implementations of
fundamental deep learning building blocks.

Quick Start::

    from pynnet.network import sequential
    from pynnet.layers import dense
    from pynnet.activation import activation, relu, relu_derivative
    from pynnet.optimizer import Adam
    from pynnet.loss import mse, mse_derivative

    model = sequential()
    model.add(dense(input_size=2, output_size=4, weight_init='he'))
    model.add(activation(relu, relu_derivative))
    model.add(dense(input_size=4, output_size=1, weight_init='xavier'))

    model.compile(loss=mse, loss_derivative=mse_derivative,
                  optimizer=Adam(learning_rate=0.01))
    model.fit(X_train, y_train, epochs=100)
"""

__version__ = "1.0.1"
__author__ = "Zain Qamar"

# Core
from .network import sequential

# Layers
from .layers import (
    dense, conv2d, maxpool2d, flatten, dropout, batchnorm, lstm
)

# Activations
from .activation import (
    activation, softmax,
    relu, relu_derivative,
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    linear, linear_derivative,
    leaky_relu, leaky_relu_derivative,
    elu, elu_derivative,
    swish, swish_derivative,
    softplus, softplus_derivative,
)

# Loss functions
from .loss import (
    mse, mse_derivative,
    mae, mae_derivative,
    huber_loss, huber_loss_derivative,
    binary_cross_entropy, binary_cross_entropy_derivative,
    categorical_cross_entropy,
    cce_derivative, cce_derivative_with_softmax,
)

# Optimizers
from .optimizer import (
    BaseOptimizer, SGD, Momentum, RMSprop, Adam,
)
