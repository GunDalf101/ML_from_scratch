"""
Support Vector Machine implementation from scratch.
"""

import numpy as np

def linear(X, W, b):
    return np.dot(X, W) + b

class LinearSVM:
    """
    Support Vector Machine model implementation.
    """
    def __init__(self, learning_rate=0.001, lambda_para=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_para = lambda_para
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self.n_samples, self.n_features = X.shape
        
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for i, x in enumerate(X):
                dw, db = (0, 0)
                if y_[i] * linear(x, self.weights, -self.bias) >= 1:
                    dw = 2 * self.lambda_para * self.weights
                else:
                    dw = (2 * self.lambda_para * self.weights) - np.dot(x, y_[i])
                    db = y_[i]
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                    
    
    def predict(self, X):
        return np.sign(linear(X, self.weights, -self.bias))