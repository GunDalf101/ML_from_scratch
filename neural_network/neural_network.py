"""
Neural Network implementation from scratch.
"""
import numpy as np

class NeuralNetwork:
    """
    Neural Network model implementation.
    """
    def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=1000, activation='sigmoid', 
                 batch_size=None, lambda_param=0, random_state=None):
        """
        Initialize the neural network.
        
        Attributes:
            layer_sizes : list
                List of integers representing the number of neurons in each layer.
                For example, [2, 5, 1] creates a network with 2 input neurons, 
                5 neurons in the hidden layer, and 1 output neuron.
            learning_rate : float, default=0.01
                Learning rate for gradient descent.
            n_iterations : int, default=1000
                Number of training iterations (epochs).
            activation : str, default='sigmoid'
                Activation function to use for hidden layers.
                Options: 'sigmoid', 'relu', 'tanh'.
            batch_size : int or None, default=None
                Size of mini-batches for gradient descent. If None, use the whole dataset.
            lambda_param : float, default=0
                L2 regularization parameter.
            random_state : int or None, default=None
                Seed for random number generation for reproducible results.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation = activation
        self.batch_size = batch_size
        self.lambda_param = lambda_param

        if random_state is not None:
            np.random.seed(random_state)
        
        self.weights = []
        self.biases = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases for each layer."""
        for i in range(1, len(self.layer_sizes)):
            self.weights.append(np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * 
                              np.sqrt(2 / self.layer_sizes[i-1]))
            self.biases.append(np.zeros((1, self.layer_sizes[i])))
    
    def _activation_function(self, x):
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def _activation_derivative(self, x):
        """Calculate derivative of activation function."""
        if self.activation == 'sigmoid':
            s = self._activation_function(x)
            return s * (1 - s)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - np.power(np.tanh(x), 2)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    
    def _forward_pass(self, X):
        """
        Perform forward pass through the network.
        """
        activations = [X] 
        layer_inputs = []

        activation = X
        for i in range(len(self.weights)):
            layer_input = np.dot(activation, self.weights[i]) + self.biases[i]
            layer_inputs.append(layer_input)

            if i == len(self.weights) - 1:
                if self.layer_sizes[-1] == 1:
                    activation = 1 / (1 + np.exp(-np.clip(layer_input, -500, 500)))
                else:
                    activation = self._softmax(layer_input)
            else:
                activation = self._activation_function(layer_input)
                
            activations.append(activation)
            
        return activations, layer_inputs
    
    def _compute_cost(self, y_pred, y_true):
        """
        Compute the cost function.
        """
        m = y_true.shape[0]
        
        if self.layer_sizes[-1] == 1:
            cost = -np.mean(y_true * np.log(y_pred + 1e-15) + 
                           (1 - y_true) * np.log(1 - y_pred + 1e-15))
        else:
            cost = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        
        if self.lambda_param > 0:
            l2_cost = 0
            for w in self.weights:
                l2_cost += np.sum(np.square(w))
            cost += (self.lambda_param / (2 * m)) * l2_cost
            
        return cost
    
    def _backward_pass(self, X, y, activations, layer_inputs):
        """
        Perform backward pass (backpropagation).
        """
        m = X.shape[0]
        
        output_error = activations[-1] - y
        
        delta = output_error
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if self.lambda_param > 0:
                dW += (self.lambda_param / m) * self.weights[i]
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(layer_inputs[i-1])
    
    def fit(self, X, y, verbose=False):
        """
        Train the neural network.
        """
        m = X.shape[0]
        
        if self.layer_sizes[-1] == 1:
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        else:
            if len(y.shape) == 1:
                y_onehot = np.zeros((m, self.layer_sizes[-1]))
                y_onehot[np.arange(m), y.astype(int)] = 1
                y = y_onehot
        
        for epoch in range(self.n_iterations):
            if self.batch_size is None:
                activations, layer_inputs = self._forward_pass(X)
                self._backward_pass(X, y, activations, layer_inputs)
            else:
                indices = np.random.permutation(m)
                for start_idx in range(0, m, self.batch_size):
                    batch_indices = indices[start_idx:min(start_idx + self.batch_size, m)]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    activations, layer_inputs = self._forward_pass(X_batch)
                    self._backward_pass(X_batch, y_batch, activations, layer_inputs)
            
            if verbose and epoch % (self.n_iterations // 10) == 0:
                activations, _ = self._forward_pass(X)
                cost = self._compute_cost(activations[-1], y)
                print(f"Epoch {epoch}/{self.n_iterations}, Cost: {cost:.4f}")
    
    def predict(self, X):
        """
        Make predictions.
        """
        activations, _ = self._forward_pass(X)
        predictions = activations[-1]
        
        if predictions.shape[1] == 1:
            return (predictions >= 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """
        Predict probability estimates.
        """
        activations, _ = self._forward_pass(X)
        return activations[-1]