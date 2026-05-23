"""
XOR Gate Neural Network Example
This example demonstrates how to create a simple neural network
that learns the XOR logic gate operation.

XOR Truth Table:
0 0 => 0
0 1 => 1
1 0 => 1
1 1 => 0
"""

import numpy as np
from pynnet.network import sequential
from pynnet.layers import dense
from pynnet.activation import activation, relu, relu_derivative, sigmoid, sigmoid_derivative
from pynnet.optimizer import Adam
from pynnet.loss import mse, mse_derivative

# Create training data
X = np.array([
    [[0, 0]],  # Input 1
    [[0, 1]],  # Input 2
    [[1, 0]],  # Input 3
    [[1, 1]]   # Input 4
])

# Create target outputs for XOR operation
y = np.array([
    [[0]],  # 0 XOR 0 = 0
    [[1]],  # 0 XOR 1 = 1
    [[1]],  # 1 XOR 0 = 1
    [[0]]   # 1 XOR 1 = 0
])

# Create the model
model = sequential()

# Add layers with appropriate initialization
model.add(dense(input_size=2, output_size=4, weight_init='he'))  # He init for ReLU
model.add(activation(relu, relu_derivative))
model.add(dense(input_size=4, output_size=1, weight_init='xavier'))  # Xavier init for sigmoid
model.add(activation(sigmoid, sigmoid_derivative))

# Compile the model
model.compile(
    loss=mse,
    loss_derivative=mse_derivative,
    optimizer=Adam(learning_rate=0.01)
)

# Train the model
print("Training the XOR model...")
model.fit(X, y, epochs=1000, verbose=True, print_every=100)

# Test the model
print("\nTesting the model:")
predictions = model.predict(X)

# Print results
print("\nPredictions:")
print("Input\t\tTarget\tPrediction")
print("-" * 40)
for i in range(len(X)):
    x = X[i][0]
    target = y[i][0][0]
    pred = predictions[i][0][0]
    print(f"{x}\t{target}\t{pred:.4f}")