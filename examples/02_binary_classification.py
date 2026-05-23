"""
Binary Classification Example
This example shows how to create a neural network for binary classification
using a simple dataset of points in 2D space.
"""

import numpy as np
from pynnet.network import sequential
from pynnet.layers import dense
from pynnet.activation import activation, relu, relu_derivative, sigmoid, sigmoid_derivative
from pynnet.optimizer import Adam
from pynnet.loss import binary_cross_entropy, binary_cross_entropy_derivative

# Generate synthetic data: two circles
def generate_circle_data(n_samples=100, noise=0.1):
    # Generate inner circle
    r1 = 0.5 + noise * np.random.randn(n_samples // 2, 1)
    theta1 = 2 * np.pi * np.random.rand(n_samples // 2, 1)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    inner_circle = np.hstack([x1, y1])
    inner_labels = np.zeros((n_samples // 2, 1))

    # Generate outer circle
    r2 = 2 + noise * np.random.randn(n_samples // 2, 1)
    theta2 = 2 * np.pi * np.random.rand(n_samples // 2, 1)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    outer_circle = np.hstack([x2, y2])
    outer_labels = np.ones((n_samples // 2, 1))

    # Combine data
    X = np.vstack([inner_circle, outer_circle])
    y = np.vstack([inner_labels, outer_labels])

    # Reshape for the network
    X = X.reshape(-1, 1, 2)
    y = y.reshape(-1, 1, 1)

    return X, y

# Generate data
X, y = generate_circle_data(n_samples=200, noise=0.1)

# Create the model
model = sequential()

# Add layers
model.add(dense(input_size=2, output_size=8, weight_init='he'))
model.add(activation(relu, relu_derivative))
model.add(dense(input_size=8, output_size=4, weight_init='he'))
model.add(activation(relu, relu_derivative))
model.add(dense(input_size=4, output_size=1, weight_init='xavier'))
model.add(activation(sigmoid, sigmoid_derivative))

# Compile model with binary cross-entropy loss
model.compile(
    loss=binary_cross_entropy,
    loss_derivative=binary_cross_entropy_derivative,
    optimizer=Adam(learning_rate=0.01)
)

# Train the model
print("Training the binary classification model...")
model.fit(X, y, epochs=500, verbose=True, print_every=50)

# Test the model
print("\nTesting the model...")
predictions = model.predict(X)

# Calculate accuracy
accuracy = np.mean((predictions > 0.5) == y)
print(f"\nAccuracy: {accuracy:.2%}")

# Print some example predictions
print("\nSample Predictions:")
print("Input Point\t\tTrue Class\tPredicted\tPrediction")
print("-" * 60)
for i in range(5):
    x = X[i][0]
    true_class = y[i][0][0]
    pred = predictions[i][0][0]
    pred_class = 1 if pred > 0.5 else 0
    print(f"({x[0]:.2f}, {x[1]:.2f})\t\t{true_class:.0f}\t\t{pred_class}\t\t{pred:.4f}")