"""
Convolutional Layer (Conv2D)
=============================

Implements a 2D convolution layer with learnable kernels and biases,
suitable for image-based tasks. Uses ``'valid'`` convolution by default
with optional zero-padding.
"""

import numpy as np
from .base import layer


class conv2d(layer):
    """2D Convolutional layer.

    Performs spatial convolution over an input volume of shape
    ``(channels_in, height, width)`` using ``num_filters`` learnable
    kernels of size ``(kernel_size, kernel_size)``.

    Args:
        num_filters (int): Number of output feature maps (channels out).
        kernel_size (int): Height and width of each square kernel.
        input_channels (int): Number of input channels (e.g., 1 for
            grayscale, 3 for RGB).
        stride (int): Step size for sliding the kernel (default: 1).
        padding (int): Number of zero-padding rows/cols added to each
            side of the input (default: 0).
        weight_init (str): Weight initialization strategy. One of
            ``'he'``, ``'xavier'``, ``'random'`` (default: ``'he'``).

    Attributes:
        weights (np.ndarray): Kernel tensor of shape
            ``(num_filters, input_channels, kernel_size, kernel_size)``.
        biases (np.ndarray): Bias vector of shape ``(num_filters, 1, 1)``.
    """

    def __init__(self, num_filters, kernel_size, input_channels,
                 stride=1, padding=0, weight_init='he'):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.stride = stride
        self.padding = padding

        # Initialize weights
        fan_in = input_channels * kernel_size * kernel_size
        fan_out = num_filters * kernel_size * kernel_size

        if weight_init == 'he':
            std = np.sqrt(2.0 / fan_in)
        elif weight_init == 'xavier':
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            std = 0.01

        self.weights = np.random.randn(
            num_filters, input_channels, kernel_size, kernel_size
        ) * std
        self.biases = np.zeros((num_filters, 1, 1))

    def _pad(self, x):
        """Apply zero-padding to the spatial dimensions.

        Args:
            x (np.ndarray): Input of shape ``(C, H, W)``.

        Returns:
            np.ndarray: Zero-padded input.
        """
        if self.padding == 0:
            return x
        return np.pad(
            x,
            ((0, 0),
             (self.padding, self.padding),
             (self.padding, self.padding)),
            mode='constant',
            constant_values=0
        )

    def forward(self, input_data):
        """Forward pass: convolve input with learned kernels.

        Args:
            input_data (np.ndarray): Input tensor of shape
                ``(input_channels, H, W)``.

        Returns:
            np.ndarray: Output feature maps of shape
                ``(num_filters, out_H, out_W)``.
        """
        self.input = input_data
        padded = self._pad(input_data)
        _, h, w = padded.shape
        kh, kw = self.kernel_size, self.kernel_size
        s = self.stride

        out_h = (h - kh) // s + 1
        out_w = (w - kw) // s + 1

        self.output = np.zeros((self.num_filters, out_h, out_w))

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = padded[
                        :,
                        i * s: i * s + kh,
                        j * s: j * s + kw
                    ]
                    self.output[f, i, j] = (
                        np.sum(region * self.weights[f]) + self.biases[f, 0, 0]
                    )

        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """Backward pass: compute gradients for kernels, biases, and input.

        Args:
            output_gradient (np.ndarray): Gradient of loss w.r.t. this
                layer's output, shape ``(num_filters, out_H, out_W)``.
            learning_rate: Unused (optimizer handles updates).

        Returns:
            np.ndarray: Gradient of loss w.r.t. this layer's input.
        """
        padded = self._pad(self.input)
        _, h, w = padded.shape
        kh, kw = self.kernel_size, self.kernel_size
        s = self.stride
        _, out_h, out_w = output_gradient.shape

        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)
        input_gradient_padded = np.zeros_like(padded)

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = padded[
                        :,
                        i * s: i * s + kh,
                        j * s: j * s + kw
                    ]
                    self.weights_gradient[f] += (
                        output_gradient[f, i, j] * region
                    )
                    self.biases_gradient[f, 0, 0] += output_gradient[f, i, j]
                    input_gradient_padded[
                        :,
                        i * s: i * s + kh,
                        j * s: j * s + kw
                    ] += output_gradient[f, i, j] * self.weights[f]

        # Remove padding from gradient if padding was applied
        if self.padding > 0:
            p = self.padding
            return input_gradient_padded[:, p:-p, p:-p]

        return input_gradient_padded
