"""
Perceptron implementation from scratch.
"""

import numpy as np

class Perceptron:
    """
    Perceptron class implementation.
    """
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_func = self.Heaviside_step_fn
        self.weights = None
        self.bias = None
    
    def linear(self, X):
        Z = np.dot(X, self.weights) + self.bias
        return Z
    
    def Heaviside_step_fn(self, x):
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y_ = [1 if i > 0 else 0 for i in y]
        
        for _ in range(self.n_iterations):
            for i, x in enumerate(X):
                prediction = self.predict(x)
                error = prediction - y[i]
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error
    
    def predict(self, X):
        Z = self.linear(X)
        pred = self.activation_func(Z)
        return pred