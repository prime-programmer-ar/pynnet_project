"""
Batch Normalization Layer
==========================

Implements batch normalization for stabilizing and accelerating
training. Since this library trains sample-by-sample, this layer
normalizes across the feature dimension using running statistics.
"""

import numpy as np
from .base import layer


class batchnorm(layer):
    """Batch Normalization layer.

    Normalizes the input to have zero mean and unit variance, then
    applies a learnable scale (``gamma``) and shift (``beta``).

    Since pynnet trains one sample at a time, this layer maintains
    running estimates of mean and variance for normalization.

    Args:
        num_features (int): Number of features (neurons / channels)
            to normalize over.
        momentum (float): Coefficient for updating running statistics.
            A value of 0.1 means ``running = 0.9 * running + 0.1 * batch``.
        epsilon (float): Small constant for numerical stability in
            the denominator.

    Attributes:
        weights (np.ndarray): Scale parameter ``gamma``, shape ``(1, num_features)``.
        biases (np.ndarray): Shift parameter ``beta``, shape ``(1, num_features)``.
        training (bool): Toggle between training and inference mode.
    """

    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters (gamma and beta)
        self.weights = np.ones((1, num_features))   # gamma
        self.biases = np.zeros((1, num_features))    # beta

        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        self.training = True

    def forward(self, input_data):
        """Forward pass: normalize, scale, and shift.

        During training, uses the current sample's statistics and
        updates running estimates. During inference, uses running
        statistics.

        Args:
            input_data (np.ndarray): Input of shape ``(1, num_features)``.

        Returns:
            np.ndarray: Normalized output, same shape.
        """
        self.input = input_data

        if self.training:
            self.mean = np.mean(input_data, axis=0, keepdims=True)
            self.var = np.var(input_data, axis=0, keepdims=True)

            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean +
                self.momentum * self.mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var +
                self.momentum * self.var
            )
        else:
            self.mean = self.running_mean
            self.var = self.running_var

        self.x_norm = (input_data - self.mean) / np.sqrt(self.var + self.epsilon)
        self.output = self.weights * self.x_norm + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """Backward pass: compute gradients for gamma, beta, and input.

        Args:
            output_gradient (np.ndarray): Gradient from upstream,
                shape ``(1, num_features)``.
            learning_rate: Unused.

        Returns:
            np.ndarray: Gradient w.r.t. input.
        """
        # Gradient of gamma and beta
        self.weights_gradient = np.sum(
            output_gradient * self.x_norm, axis=0, keepdims=True
        )
        self.biases_gradient = np.sum(
            output_gradient, axis=0, keepdims=True
        )

        # Gradient of input (simplified for single-sample batches)
        dx_norm = output_gradient * self.weights
        inv_std = 1.0 / np.sqrt(self.var + self.epsilon)
        return dx_norm * inv_std
