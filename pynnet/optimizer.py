import numpy as np

class BaseOptimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, learning_rate=0.01):
        """
        Initializes the optimizer.
        
        Args:
            learning_rate (float): The step size for parameter updates.
        """
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        This method will be called for each layer to update its parameters.
        It must be implemented by subclasses.
        """
        raise NotImplementedError

    def step(self):
        """
        Called once per training batch (after all gradients are computed).
        Used by optimizers like Adam that require a timestep counter.
        """
        pass # Most optimizers don't need this

# --- Standard Optimizer ---

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.
    The simplest update rule: param = param - lr * gradient
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, layer):
        """
        Updates the parameters of a single layer using standard SGD.
        """
        # Check if the layer is trainable (i.e., has weights)
        if hasattr(layer, 'weights'):
            layer.weights -= self.learning_rate * layer.weights_gradient
            layer.biases -= self.learning_rate * layer.biases_gradient

# --- Adaptive Optimizers ---

class Momentum(BaseOptimizer):
    """
    Stochastic Gradient Descent with Momentum.
    Helps accelerate SGD in the relevant direction and dampens oscillations.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        # Stores the "velocity" for each parameter
        self.velocities = {}

    def update(self, layer):
        """
        Updates the parameters of a single layer using Momentum.
        """
        if hasattr(layer, 'weights'):
            layer_id = id(layer)
            
            # Initialize velocities if this is the first time seeing this layer
            if layer_id not in self.velocities:
                self.velocities[layer_id] = {
                    'w': np.zeros_like(layer.weights),
                    'b': np.zeros_like(layer.biases)
                }

            # --- Update Weight Velocity ---
            v_w = (self.momentum * self.velocities[layer_id]['w'] -
                   self.learning_rate * layer.weights_gradient)
            
            # --- Update Bias Velocity ---
            v_b = (self.momentum * self.velocities[layer_id]['b'] -
                   self.learning_rate * layer.biases_gradient)
            
            # --- Update Parameters ---
            layer.weights += v_w
            layer.biases += v_b
            
            # --- Store new velocities for next iteration ---
            self.velocities[layer_id]['w'] = v_w
            self.velocities[layer_id]['b'] = v_b

class RMSprop(BaseOptimizer):
    """
    Root Mean Square Propagation (RMSprop) optimizer.
    Adapts the learning rate per-parameter, dividing by a running average
    of the magnitudes of recent gradients.
    """
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = rho # Decay rate for the moving average
        self.epsilon = epsilon # Small constant to prevent division by zero
        # Stores the running average of squared gradients
        self.v = {}

    def update(self, layer):
        """
        Updates the parameters of a single layer using RMSprop.
        """
        if hasattr(layer, 'weights'):
            layer_id = id(layer)
            
            if layer_id not in self.v:
                self.v[layer_id] = {
                    'w': np.zeros_like(layer.weights),
                    'b': np.zeros_like(layer.biases)
                }
            
            g_w = layer.weights_gradient
            g_b = layer.biases_gradient
            
            # --- Update Weight Velocity (v) ---
            self.v[layer_id]['w'] = (self.rho * self.v[layer_id]['w'] +
                                    (1 - self.rho) * (g_w ** 2))
            
            # --- Update Bias Velocity (v) ---
            self.v[layer_id]['b'] = (self.rho * self.v[layer_id]['b'] +
                                    (1 - self.rho) * (g_b ** 2))
            
            # --- Update Parameters ---
            layer.weights -= (self.learning_rate * g_w / 
                              (np.sqrt(self.v[layer_id]['w']) + self.epsilon))
            layer.biases -= (self.learning_rate * g_b /
                             (np.sqrt(self.v[layer_id]['b']) + self.epsilon))

class Adam(BaseOptimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    Combines ideas from Momentum (first moment) and RMSprop (second moment).
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {} # First moment (like momentum)
        self.v = {} # Second moment (like RMSprop)
        self.t = 0  # Timestep counter for bias correction

    def step(self):
        """
        Called once per batch. Increments the timestep.
        """
        self.t += 1

    def update(self, layer):
        """
        Updates the parameters of a single layer using Adam.
        """
        if hasattr(layer, 'weights'):
            layer_id = id(layer)

            if layer_id not in self.m:
                self.m[layer_id] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}
                self.v[layer_id] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}
            
            g_w = layer.weights_gradient
            g_b = layer.biases_gradient

            # --- Update First Moment (m) ---
            self.m[layer_id]['w'] = self.beta1 * self.m[layer_id]['w'] + (1 - self.beta1) * g_w
            self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * g_b

            # --- Update Second Moment (v) ---
            self.v[layer_id]['w'] = self.beta2 * self.v[layer_id]['w'] + (1 - self.beta2) * (g_w ** 2)
            self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * (g_b ** 2)

            # --- Bias Correction ---
            m_w_hat = self.m[layer_id]['w'] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m[layer_id]['b'] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v[layer_id]['w'] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v[layer_id]['b'] / (1 - self.beta2 ** self.t)

            # --- Update Parameters ---
            layer.weights -= (self.learning_rate * m_w_hat / 
                              (np.sqrt(v_w_hat) + self.epsilon))
            layer.biases -= (self.learning_rate * m_b_hat /
                             (np.sqrt(v_b_hat) + self.epsilon))