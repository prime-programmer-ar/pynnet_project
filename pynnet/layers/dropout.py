"""
Dropout Layer
==============

Implements inverted dropout for regularization during training.
During inference, the layer acts as a pass-through.
"""

import numpy as np
from .base import layer


class dropout(layer):
    """Dropout regularization layer (inverted dropout).

    Randomly sets a fraction ``rate`` of input values to zero during
    training. The remaining values are scaled up by ``1 / (1 - rate)``
    so that the expected sum is preserved.

    During inference (when ``training=False``), the layer is a
    pass-through.

    Args:
        rate (float): Fraction of input units to drop. Must be in
            ``[0, 1)``. A value of 0.0 means no dropout.

    Attributes:
        training (bool): Set to ``True`` during training, ``False``
            during inference. Defaults to ``True``.
    """

    def __init__(self, rate=0.5):
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        self.training = True
        self._mask = None

    def forward(self, input_data):
        """Forward pass: apply dropout mask if training.

        Args:
            input_data (np.ndarray): Input tensor of any shape.

        Returns:
            np.ndarray: Masked (and scaled) output during training,
                or unmodified input during inference.
        """
        if self.training and self.rate > 0:
            self._mask = (
                np.random.binomial(1, 1.0 - self.rate, size=input_data.shape)
                / (1.0 - self.rate)
            )
            return input_data * self._mask
        return input_data

    def backward(self, output_gradient, learning_rate=None):
        """Backward pass: apply the same mask used in forward.

        Args:
            output_gradient (np.ndarray): Gradient from upstream.
            learning_rate: Unused.

        Returns:
            np.ndarray: Masked gradient.
        """
        if self.training and self.rate > 0:
            return output_gradient * self._mask
        return output_gradient
