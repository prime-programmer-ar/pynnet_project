"""
Flatten Layer
==============

Reshapes a multi-dimensional tensor into a 2D matrix suitable for
fully-connected (dense) layers.
"""

import numpy as np
from .base import layer


class flatten(layer):
    """Flatten layer — reshapes ``(C, H, W)`` to ``(1, C*H*W)``.

    This layer bridges convolutional/pooling layers and dense layers
    by collapsing all spatial dimensions into a single feature vector.

    Note:
        The output shape ``(1, n_features)`` uses a leading batch
        dimension of 1 to match the library's sample-by-sample
        training convention.
    """

    def forward(self, input_data):
        """Forward pass: flatten the input.

        Args:
            input_data (np.ndarray): Input of shape ``(C, H, W)`` or
                any multi-dimensional shape.

        Returns:
            np.ndarray: Flattened output of shape ``(1, n_features)``.
        """
        self.input_shape = input_data.shape
        return input_data.reshape(1, -1)

    def backward(self, output_gradient, learning_rate=None):
        """Backward pass: reshape gradient back to original shape.

        Args:
            output_gradient (np.ndarray): Gradient of shape
                ``(1, n_features)``.
            learning_rate: Unused.

        Returns:
            np.ndarray: Gradient reshaped to original input shape.
        """
        return output_gradient.reshape(self.input_shape)
