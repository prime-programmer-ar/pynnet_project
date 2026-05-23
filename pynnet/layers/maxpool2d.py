"""
Max Pooling Layer (MaxPool2D)
==============================

Implements 2D max pooling that reduces spatial dimensions by taking
the maximum value within non-overlapping (or strided) windows.
"""

import numpy as np
from .base import layer


class maxpool2d(layer):
    """2D Max Pooling layer.

    Reduces spatial dimensions by taking the maximum value in each
    ``(pool_size, pool_size)`` window.

    Args:
        pool_size (int): Height and width of the pooling window
            (default: 2).
        stride (int, optional): Step size. Defaults to ``pool_size``
            (non-overlapping windows).

    Note:
        During backpropagation, gradients are routed only to the
        positions that held the maximum value in each window.
    """

    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, input_data):
        """Forward pass: apply max pooling.

        Args:
            input_data (np.ndarray): Input of shape ``(C, H, W)``.

        Returns:
            np.ndarray: Pooled output of shape ``(C, out_H, out_W)``.
        """
        self.input = input_data
        c, h, w = input_data.shape
        ps = self.pool_size
        s = self.stride

        out_h = (h - ps) // s + 1
        out_w = (w - ps) // s + 1

        self.output = np.zeros((c, out_h, out_w))

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    region = input_data[
                        ch,
                        i * s: i * s + ps,
                        j * s: j * s + ps
                    ]
                    self.output[ch, i, j] = np.max(region)

        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """Backward pass: route gradients to max-value positions.

        Args:
            output_gradient (np.ndarray): Gradient of shape
                ``(C, out_H, out_W)``.
            learning_rate: Unused.

        Returns:
            np.ndarray: Gradient of shape ``(C, H, W)``.
        """
        input_gradient = np.zeros_like(self.input)
        c, out_h, out_w = output_gradient.shape
        ps = self.pool_size
        s = self.stride

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    region = self.input[
                        ch,
                        i * s: i * s + ps,
                        j * s: j * s + ps
                    ]
                    # Create mask where max value was
                    max_val = np.max(region)
                    mask = (region == max_val)
                    input_gradient[
                        ch,
                        i * s: i * s + ps,
                        j * s: j * s + ps
                    ] += output_gradient[ch, i, j] * mask

        return input_gradient
