"""
Sequential Neural Network
===========================

The core model class that orchestrates forward passes, backpropagation,
optimizer updates, and model persistence.
"""

import numpy as np
import time
from .optimizer import SGD


class sequential:
    """A sequential model — a linear stack of layers.

    Layers are executed in order during the forward pass and in
    reverse order during backpropagation.

    Example::

        from pynnet.network import sequential
        from pynnet.layers import dense
        from pynnet.activation import activation, relu, relu_derivative

        model = sequential()
        model.add(dense(input_size=2, output_size=4))
        model.add(activation(relu, relu_derivative))
    """

    def __init__(self):
        """Initialize an empty sequential model."""
        self.layers = []
        self.loss = None
        self.loss_derivative = None
        self.optimizer = None

    def add(self, layer):
        """Append a layer to the model.

        Args:
            layer: An instance of a Layer class (e.g., dense, activation).
        """
        self.layers.append(layer)

    def compile(self, loss, loss_derivative, optimizer):
        """Configure the model for training.

        Args:
            loss (callable): Loss function with signature ``(y_true, y_pred) -> float``.
            loss_derivative (callable): Derivative of the loss with signature
                ``(y_true, y_pred) -> np.ndarray``.
            optimizer (BaseOptimizer): An optimizer instance (e.g., ``Adam()``).
        """
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.optimizer = optimizer
        print("Model compiled successfully.")

    def set_loss(self, loss_function, loss_derivative):
        """Set the loss function (legacy method).

        .. deprecated::
            Use :meth:`compile` instead — it also sets the optimizer.
        """
        self.loss = loss_function
        self.loss_derivative = loss_derivative
        if self.optimizer is None:
            self.optimizer = SGD()
            print("Warning: `set_loss` is deprecated. Use `compile`. "
                  "Defaulting to SGD optimizer.")

    def predict(self, input_data):
        """Generate predictions for input samples.

        Args:
            input_data (np.ndarray): Input data of shape
                ``(num_samples, ...)``.

        Returns:
            np.ndarray: Array of predictions.
        """
        num_samples = len(input_data)
        results = []

        for i in range(num_samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            results.append(output)

        return np.array(results)

    def fit(self, x_train, y_train, epochs, learning_rate=None,
            verbose=True, print_every=100):
        """Train the model for a fixed number of epochs (online / SGD-style).

        Each sample is fed individually (stochastic gradient descent).
        After every sample the optimizer updates all trainable parameters.

        Args:
            x_train (np.ndarray): Training inputs of shape
                ``(num_samples, ...)``.
            y_train (np.ndarray): Training targets of shape
                ``(num_samples, ...)``.
            epochs (int): Number of full passes over the dataset.
            learning_rate (float, optional): If provided, overrides the
                optimizer's current learning rate.
            verbose (bool): Whether to print progress.
            print_every (int): Print frequency (in epochs).

        Raises:
            ValueError: If the model has not been compiled.
        """
        # Pre-training checks
        if self.optimizer is None:
            raise ValueError(
                "Model must be compiled before fitting. Call model.compile(...)"
            )
        if self.loss is None or self.loss_derivative is None:
            raise ValueError(
                "Loss function is not set. Call model.compile(...)"
            )

        if learning_rate is not None:
            self.optimizer.learning_rate = learning_rate

        num_samples = len(x_train)
        if num_samples == 0:
            print("Warning: Training data is empty.")
            return

        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        average_error = float('inf')

        # --- Training Loop ---
        for epoch in range(epochs):
            total_error = 0.0

            for i in range(num_samples):
                x_sample = x_train[i]
                y_sample = y_train[i]

                # 1. Forward pass
                output = x_sample
                for layer in self.layers:
                    output = layer.forward(output)

                # 2. Compute loss
                total_error += self.loss(y_sample, output)

                # 3. Backward pass
                gradient = self.loss_derivative(y_sample, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, None)

                # 4. Optimizer step
                self.optimizer.step()
                for layer in self.layers:
                    self.optimizer.update(layer)

            # --- Epoch reporting ---
            average_error = total_error / num_samples

            if verbose and (epoch == 0 or
                           (epoch + 1) % print_every == 0 or
                           (epoch + 1) == epochs):
                elapsed = time.time() - start_time
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Error: {average_error:.6f} | "
                      f"Time: {elapsed:.2f}s")

        print(f"Training complete. Final Error: {average_error:.6f}")

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def get_parameters(self):
        """Retrieve all trainable parameters.

        Returns:
            list[tuple]: List of ``(weights, biases)`` tuples for each
            trainable layer, in order.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append((layer.weights, layer.biases))
        return params

    def set_parameters(self, params):
        """Load parameters into the model's trainable layers.

        Args:
            params (list[tuple]): List of ``(weights, biases)`` tuples.

        Raises:
            ValueError: If parameter count or shapes don't match.
        """
        param_iter = iter(params)
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                try:
                    weights, biases = next(param_iter)
                except StopIteration:
                    raise ValueError(
                        "Not enough parameter sets to load. "
                        "Model architecture does not match saved weights."
                    )
                if layer.weights.shape != weights.shape:
                    raise ValueError(
                        f"Shape mismatch for weights in {type(layer).__name__}. "
                        f"Expected {layer.weights.shape}, got {weights.shape}."
                    )
                if layer.biases.shape != biases.shape:
                    raise ValueError(
                        f"Shape mismatch for biases in {type(layer).__name__}. "
                        f"Expected {layer.biases.shape}, got {biases.shape}."
                    )
                layer.weights = weights
                layer.biases = biases

        # Check for leftover params
        try:
            next(param_iter)
            raise ValueError(
                "Too many parameter sets to load. "
                "Model architecture does not match saved weights."
            )
        except StopIteration:
            pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, filepath):
        """Save model parameters to a ``.npz`` file.

        Args:
            filepath (str): Destination path (e.g., ``'my_model.npz'``).
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'

        params = self.get_parameters()
        if not params:
            print("Warning: Model has no trainable parameters to save.")
            return

        param_dict = {}
        for i, (w, b) in enumerate(params):
            param_dict[f'layer_{i}_weights'] = w
            param_dict[f'layer_{i}_biases'] = b

        np.savez(filepath, **param_dict)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load model parameters from a ``.npz`` file.

        The model must have the **exact same architecture** as when the
        weights were saved.

        Args:
            filepath (str): Path to the ``.npz`` weights file.
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'

        try:
            data = np.load(filepath)
        except FileNotFoundError:
            print(f"Error: No weights file found at {filepath}")
            return
        except Exception as e:
            print(f"Error loading weights file: {e}")
            return

        params = []
        i = 0
        while f'layer_{i}_weights' in data:
            if f'layer_{i}_biases' not in data:
                print(f"Error: Weights file is corrupt. "
                      f"Missing biases for layer {i}.")
                return
            params.append((data[f'layer_{i}_weights'],
                           data[f'layer_{i}_biases']))
            i += 1

        if not params:
            print("Error: No layer parameters found in file.")
            return

        try:
            self.set_parameters(params)
            print(f"Model weights loaded successfully from {filepath}")
        except ValueError as e:
            print(f"Error loading weights: {e}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self):
        """Print a summary of the model architecture.

        Shows each layer's type, output shape (if known), and
        parameter count.
        """
        print("=" * 60)
        print(f"{'Layer (type)':<30} {'Params':>10}")
        print("=" * 60)
        total_params = 0
        for i, layer in enumerate(self.layers):
            name = type(layer).__name__
            params = 0
            if hasattr(layer, 'weights'):
                params += layer.weights.size + layer.biases.size
            total_params += params
            print(f"{i}: {name:<27} {params:>10,}")
        print("=" * 60)
        print(f"Total trainable parameters: {total_params:,}")
        print("=" * 60)