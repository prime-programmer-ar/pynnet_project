"""
Activation Functions Module
============================

This module provides activation function layers and standalone activation
functions with their derivatives for use in neural networks.

Each activation function has a corresponding ``_derivative`` function
that computes the local gradient needed during backpropagation.
"""

import numpy as np
from .layers.base import layer


# ---------------------------------------------------------------------------
# Base Activation Layer Class
# ---------------------------------------------------------------------------

class activation(layer):
    """An activation layer that applies an element-wise activation function.

    Args:
        activation_fn (callable): The activation function to apply.
        activation_derivative_fn (callable): The derivative of the activation function.
    """

    def __init__(self, activation_fn, activation_derivative_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.activation_derivative_fn = activation_derivative_fn

    def forward(self, input_data):
        """Forward pass: apply the activation function element-wise.

        Args:
            input_data (np.ndarray): Input tensor of any shape.

        Returns:
            np.ndarray: Activated output, same shape as input.
        """
        self.input = input_data
        self.output = self.activation_fn(input_data)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """Backward pass: compute dL/dX = dL/dY * f'(X).

        Args:
            output_gradient (np.ndarray): Gradient from the layer above.
            learning_rate: Unused (no trainable parameters).

        Returns:
            np.ndarray: Gradient w.r.t. the input.
        """
        return output_gradient * self.activation_derivative_fn(self.input)


# ---------------------------------------------------------------------------
# Softmax Layer (Special Case)
# ---------------------------------------------------------------------------

class softmax(layer):
    """Softmax activation layer.

    This is implemented as a separate class because its backward pass
    is tightly coupled with Categorical Cross-Entropy loss.

    When used with ``cce_derivative_with_softmax`` as the loss derivative,
    the gradient passed into ``backward()`` is already the simplified
    ``(y_pred - y_true)`` form, so this layer simply passes it through.
    """

    def forward(self, input_data):
        """Forward pass: compute softmax probabilities.

        Uses the ``max-subtraction`` trick for numerical stability.

        Args:
            input_data (np.ndarray): Raw logits.

        Returns:
            np.ndarray: Probability distribution (sums to 1 along last axis).
        """
        # Subtract max for numerical stability (prevents exp overflow)
        shifted = input_data - np.max(input_data, axis=-1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """Backward pass for Softmax + CCE combined.

        Assumes the loss derivative has already computed the simplified
        gradient ``(y_pred - y_true) / batch_size``. This layer acts as
        a pass-through in that case.

        Args:
            output_gradient (np.ndarray): Pre-computed combined gradient.
            learning_rate: Unused.

        Returns:
            np.ndarray: The gradient unchanged.
        """
        return output_gradient


# ---------------------------------------------------------------------------
# Standalone Activation Functions & Derivatives
# ---------------------------------------------------------------------------

def linear(x):
    """Linear (identity) activation: f(x) = x."""
    return x


def linear_derivative(x):
    """Derivative of linear activation: f'(x) = 1."""
    return np.ones_like(x)


# --- Sigmoid ---

def sigmoid(x):
    """Numerically stable sigmoid activation.

    Uses a piecewise formulation to avoid overflow in ``np.exp``:
    - For x >= 0: 1 / (1 + exp(-x))
    - For x <  0: exp(x) / (1 + exp(x))

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Values in (0, 1).
    """
    result = np.zeros_like(x, dtype=np.float64)
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    # Stable for positive x
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

    # Stable for negative x (avoids exp of large positive number)
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1.0 + exp_x)

    return result


def sigmoid_derivative(x):
    """Derivative of sigmoid: f'(x) = σ(x) · (1 − σ(x)).

    Args:
        x (np.ndarray): Input tensor (pre-activation values).

    Returns:
        np.ndarray: Local gradient.
    """
    s = sigmoid(x)
    return s * (1.0 - s)


# --- Tanh ---

def tanh(x):
    """Hyperbolic tangent activation: f(x) = tanh(x).

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Values in (-1, 1).
    """
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh: f'(x) = 1 − tanh²(x).

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Local gradient.
    """
    return 1.0 - np.tanh(x) ** 2


# --- ReLU ---

def relu(x):
    """Rectified Linear Unit: f(x) = max(0, x).

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Activated output.
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU: f'(x) = 1 if x > 0, else 0.

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Local gradient (binary mask).
    """
    return np.where(x > 0, 1.0, 0.0)


# --- Leaky ReLU ---

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: f(x) = x if x > 0, else alpha * x.

    Args:
        x (np.ndarray): Input tensor.
        alpha (float): Slope for negative values (default: 0.01).

    Returns:
        np.ndarray: Activated output.
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU.

    Args:
        x (np.ndarray): Input tensor.
        alpha (float): Slope for negative values (default: 0.01).

    Returns:
        np.ndarray: Local gradient.
    """
    return np.where(x > 0, 1.0, alpha)


# --- ELU ---

def elu(x, alpha=1.0):
    """Exponential Linear Unit: f(x) = x if x > 0, else alpha * (exp(x) - 1).

    Args:
        x (np.ndarray): Input tensor.
        alpha (float): Scale for negative values (default: 1.0).

    Returns:
        np.ndarray: Activated output.
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x, alpha=1.0):
    """Derivative of ELU.

    Args:
        x (np.ndarray): Input tensor.
        alpha (float): Scale for negative values (default: 1.0).

    Returns:
        np.ndarray: Local gradient.
    """
    return np.where(x > 0, 1.0, alpha * np.exp(x))


# --- Swish ---

def swish(x):
    """Swish activation: f(x) = x · σ(x).

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Activated output.
    """
    return x * sigmoid(x)


def swish_derivative(x):
    """Derivative of Swish: f'(x) = σ(x) + x · σ(x) · (1 − σ(x)).

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Local gradient.
    """
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


# --- Softplus ---

def softplus(x):
    """Softplus activation: f(x) = log(1 + exp(x)).

    Uses a numerically stable formulation that avoids overflow
    for large positive values of x.

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Activated output (always positive).
    """
    # For large x, softplus(x) ≈ x; for small x, compute normally
    safe_x = np.where(x > 20, 0.0, x)  # avoid overflow in exp
    return np.where(x > 20, x, np.log1p(np.exp(safe_x)))


def softplus_derivative(x):
    """Derivative of softplus: f'(x) = σ(x).

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Local gradient.
    """
    return sigmoid(x)