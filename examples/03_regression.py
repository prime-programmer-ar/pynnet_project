"""
Regression Example
This example demonstrates how to use PyNNet for regression problems
by learning to predict points on a sine wave with added noise.
"""

import numpy as np
from pynnet.network import sequential
from pynnet.layers import dense
from pynnet.activation import activation, relu, relu_derivative, linear, linear_derivative
from pynnet.optimizer import Adam
from pynnet.loss import mse, mse_derivative

# Generate synthetic data
np.random.seed(42)

# Generate x values
X = np.linspace(0, 4*np.pi, 200)
# Generate y values (sine wave with noise)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Reshape data for the network
X = X.reshape(-1, 1, 1)
y = y.reshape(-1, 1, 1)

# Create the model
model = sequential()

# Add layers
# Note: We use linear activation in the output layer for regression
model.add(dense(input_size=1, output_size=32, weight_init='he'))
model.add(activation(relu, relu_derivative))
model.add(dense(input_size=32, output_size=16, weight_init='he'))
model.add(activation(relu, relu_derivative))
model.add(dense(input_size=16, output_size=1, weight_init='he'))
model.add(activation(linear, linear_derivative))  # Linear activation for regression output

# Compile the model
model.compile(
    loss=mse,
    loss_derivative=mse_derivative,
    optimizer=Adam(learning_rate=0.001)
)

# Train the model
print("Training the regression model...")
model.fit(X, y, epochs=200, verbose=True, print_every=20)

# Generate predictions
predictions = model.predict(X)

# Calculate R-squared score
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - predictions) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"\nR-squared Score: {r2_score:.4f}")

# Print some example predictions
print("\nSample Predictions:")
print("Input (x)\tTrue Value\tPredicted\tError")
print("-" * 50)
for i in range(5):
    x = X[i][0][0]
    true_val = y[i][0][0]
    pred = predictions[i][0][0]
    error = abs(true_val - pred)
    print(f"{x:.2f}\t\t{true_val:.4f}\t{pred:.4f}\t{error:.4f}")

# Optional: If matplotlib is installed, plot the results
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X.flatten(), y.flatten(), c='blue', alpha=0.5, label='True Data')
    plt.plot(X.flatten(), predictions.flatten(), 'r-', label='Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sine Wave Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
except ImportError:
    print("\nInstall matplotlib to visualize the results!")