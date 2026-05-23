"""
Dense (Fully Connected) Layer
==============================

Implements a standard fully-connected neural network layer with
configurable weight and bias initialization strategies.
"""

import numpy as np
from .base import layer


class dense(layer):
    """Fully connected (dense) layer: ``output = input @ weights + biases``.

    Supports multiple weight initialization strategies optimized for
    different activation functions.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output neurons.
        weight_init (str): Weight initialization method. One of
            ``'he'``, ``'xavier'``, ``'lecun'``, ``'orthogonal'``,
            ``'identity'``, ``'random'``.
        bias_init (str): Bias initialization method. One of
            ``'zeros'``, ``'ones'``, ``'random'``, ``'constant'``.
        bias_constant (float): Value to use when ``bias_init='constant'``.

    Attributes:
        weights (np.ndarray): Weight matrix of shape ``(input_size, output_size)``.
        biases (np.ndarray): Bias vector of shape ``(1, output_size)``.
        weights_gradient (np.ndarray): Gradient of loss w.r.t. weights (set during backward).
        biases_gradient (np.ndarray): Gradient of loss w.r.t. biases (set during backward).
    """

    def __init__(self, input_size, output_size, weight_init='he',
                 bias_init='zeros', bias_constant=0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self._initialize_weights(weight_init)
        self.biases = self._initialize_biases(bias_init, bias_constant)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_weights(self, method):
        """Initialize weight matrix using the specified strategy.

        Args:
            method (str): Initialization method name.

        Returns:
            np.ndarray: Initialized weight matrix of shape ``(input_size, output_size)``.

        Raises:
            ValueError: If *method* is not recognized or incompatible.
        """
        n_in, n_out = self.input_size, self.output_size

        if method == 'random':
            return np.random.randn(n_in, n_out) * 0.01

        elif method == 'he':
            # He (Kaiming) — optimal for ReLU
            return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

        elif method == 'xavier':
            # Glorot — optimal for sigmoid / tanh
            return np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))

        elif method == 'lecun':
            return np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)

        elif method == 'identity':
            if n_in != n_out:
                raise ValueError(
                    "Identity initialization requires input_size == output_size"
                )
            return np.eye(n_in)

        elif method == 'orthogonal':
            # Use SVD for correctness with rectangular matrices
            random_matrix = np.random.randn(n_in, n_out)
            u, _, vt = np.linalg.svd(random_matrix, full_matrices=False)
            # u: (n_in, min(n_in,n_out)), vt: (min(n_in,n_out), n_out)
            if n_in >= n_out:
                return u[:, :n_out]
            else:
                return vt[:n_in, :]

        else:
            raise ValueError(f"Unsupported weight initialization method: '{method}'")

    def _initialize_biases(self, method, constant=0.0):
        """Initialize bias vector using the specified strategy.

        Args:
            method (str): Initialization method name.
            constant (float): Value for ``'constant'`` initialization.

        Returns:
            np.ndarray: Initialized bias vector of shape ``(1, output_size)``.

        Raises:
            ValueError: If *method* is not recognized.
        """
        if method == 'zeros':
            return np.zeros((1, self.output_size))
        elif method == 'ones':
            return np.ones((1, self.output_size))
        elif method == 'random':
            return np.random.randn(1, self.output_size) * 0.01
        elif method == 'constant':
            return np.full((1, self.output_size), constant)
        else:
            raise ValueError(f"Unsupported bias initialization method: '{method}'")

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def forward(self, input_data):
        """Forward pass: ``Y = X @ W + b``.

        Args:
            input_data (np.ndarray): Input tensor of shape ``(batch, input_size)``.

        Returns:
            np.ndarray: Output tensor of shape ``(batch, output_size)``.
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """Backward pass: compute and store parameter gradients.

        Computes:
        - ``dL/dW = X^T @ dL/dY``
        - ``dL/db = sum(dL/dY, axis=0)``
        - ``dL/dX = dL/dY @ W^T``  (returned for upstream layers)

        Args:
            output_gradient (np.ndarray): Gradient of loss w.r.t. this layer's output.
            learning_rate: Unused; kept for API compatibility.

        Returns:
            np.ndarray: Gradient of loss w.r.t. this layer's input.
        """
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)
