"""
Loss Functions Module
======================

This module provides loss functions and their derivatives for training
neural networks. Each loss function has a corresponding ``_derivative``
function that computes the gradient of the loss w.r.t. the predicted output.

All functions follow the signature ``(y_true, y_pred) -> scalar_or_gradient``.
"""

import numpy as np

# Small constant to avoid log(0) and division by zero
_EPSILON = 1e-15


# ---------------------------------------------------------------------------
# Regression Loss Functions
# ---------------------------------------------------------------------------

def mse(y_true, y_pred):
    """Mean Squared Error loss.

    .. math:: L = \\frac{1}{n} \\sum (y_{true} - y_{pred})^2

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values (same shape as y_true).

    Returns:
        float: Scalar loss value.
    """
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    """Derivative of Mean Squared Error w.r.t. y_pred.

    .. math:: \\frac{\\partial L}{\\partial y_{pred}} = \\frac{2(y_{pred} - y_{true})}{n}

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Gradient, same shape as inputs.
    """
    return 2.0 * (y_pred - y_true) / y_true.size


def mae(y_true, y_pred):
    """Mean Absolute Error loss.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Scalar loss value.
    """
    return np.mean(np.abs(y_true - y_pred))


def mae_derivative(y_true, y_pred):
    """Derivative of Mean Absolute Error w.r.t. y_pred.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Gradient, same shape as inputs.
    """
    return np.sign(y_pred - y_true) / y_true.size


def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss — robust regression loss, less sensitive to outliers than MSE.

    Behaves like MSE for small errors and MAE for large errors.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        delta (float): Threshold at which to switch from quadratic to linear.

    Returns:
        float: Scalar loss value.
    """
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)


def huber_loss_derivative(y_true, y_pred, delta=1.0):
    """Derivative of Huber loss w.r.t. y_pred.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        delta (float): Threshold parameter.

    Returns:
        np.ndarray: Gradient, same shape as inputs.
    """
    error = y_pred - y_true
    return np.where(
        np.abs(error) <= delta,
        error,
        delta * np.sign(error)
    ) / y_true.size


# ---------------------------------------------------------------------------
# Classification Loss Functions
# ---------------------------------------------------------------------------

def binary_cross_entropy(y_true, y_pred):
    """Binary Cross-Entropy loss for binary classification.

    Assumes a single sigmoid output in [0, 1].

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities in (0, 1).

    Returns:
        float: Scalar loss value.
    """
    y_pred_clipped = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    return -np.mean(
        y_true * np.log(y_pred_clipped) +
        (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
    )


def binary_cross_entropy_derivative(y_true, y_pred):
    """Derivative of Binary Cross-Entropy w.r.t. y_pred.

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Gradient, same shape as inputs.
    """
    y_pred_clipped = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    return (
        (y_pred_clipped - y_true) /
        (y_pred_clipped * (1.0 - y_pred_clipped) + _EPSILON)
    ) / y_true.size


def categorical_cross_entropy(y_true, y_pred):
    """Categorical Cross-Entropy loss for multi-class classification.

    Assumes ``y_true`` is one-hot encoded and ``y_pred`` contains
    softmax probabilities.

    Args:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted probabilities from softmax.

    Returns:
        float: Scalar loss value.
    """
    y_pred_clipped = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    # Sum over classes, then mean over batch
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=-1))


def cce_derivative(y_true, y_pred):
    """Derivative of Categorical Cross-Entropy w.r.t. y_pred.

    This is the raw derivative ``-y_true / y_pred``, which is
    numerically less stable. Prefer ``cce_derivative_with_softmax``
    when the final layer is Softmax.

    Args:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Gradient, same shape as inputs.
    """
    y_pred_clipped = np.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    return (-y_true / y_pred_clipped) / y_true.shape[0]


def cce_derivative_with_softmax(y_true, y_pred):
    """Combined Softmax + CCE derivative (simplified form).

    When the final layer is Softmax and the loss is Categorical
    Cross-Entropy, the gradient simplifies to ``(y_pred - y_true) / n``.
    This avoids computing the full Jacobian of Softmax and is
    numerically stable.

    Use this as ``loss_derivative`` in ``model.compile()`` when your
    last layer is a ``softmax`` activation.

    Args:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted softmax probabilities.

    Returns:
        np.ndarray: Simplified gradient.
    """
    return (y_pred - y_true) / y_true.shape[0]