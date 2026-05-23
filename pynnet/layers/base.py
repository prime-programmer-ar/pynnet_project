class layer:
    """
    Abstract base class for all the layers in the network.
    Every layer must implement a forward pass and a backward pass.
    """
    def __init__(self):
        """
        Initialize the layer with default parameters.
        'self.input' and 'self.output' are caches for the values
        during the forward pass, which are needed for the backward pass.
        """
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        """
        Computes the output of the layer for a given input.
        This method must be implemented by subclasses.
        """
        # Note: This is an abstract method, it should be overridden in subclasses.
        raise NotImplementedError("Forward method not implemented.")
    
    def backward(self, output_gradient, learning_rate):
        """
        Computes the gradient of the loss with respect to the input of the layer, 
        and any updates to trainable parameters like weights and biases.
        this method must be implemented by subclasses.
        """
        # Note: This is an abstract method, it should be overridden in subclasses.
        raise NotImplementedError("Backward method not implemented.")