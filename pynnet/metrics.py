"""
Metrics Module
==============

This module provides common evaluation metrics for neural networks.
"""

import numpy as np


def accuracy(y_true, y_pred, threshold=0.5):
    """Compute the accuracy for binary or categorical predictions.

    If inputs are 1D/2D with a single feature column, it assumes binary classification
    and applies a threshold. If inputs have multiple feature columns, it assumes
    categorical classification and uses argmax.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted probabilities or values.
        threshold (float): Threshold for binary classification (default: 0.5).

    Returns:
        float: Accuracy score between 0.0 and 1.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle PyNNet's 3D shapes (num_samples, 1, features)
    if y_true.ndim == 3 and y_true.shape[1] == 1:
        y_true = y_true.reshape(y_true.shape[0], y_true.shape[2])
    if y_pred.ndim == 3 and y_pred.shape[1] == 1:
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2])

    if y_true.ndim == 1 or y_true.shape[-1] == 1:
        # Binary classification
        pred_labels = (y_pred >= threshold).astype(int)
        true_labels = y_true.astype(int)
        return np.mean(pred_labels == true_labels)
    else:
        # Categorical classification (one-hot encoded)
        pred_labels = np.argmax(y_pred, axis=-1)

        # Check if y_true is one-hot or integer labels
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            true_labels = np.argmax(y_true, axis=-1)
        else:
            true_labels = y_true

        return np.mean(pred_labels == true_labels)
