"""
Linear Regression implementation from scratch.
"""
import numpy as np
from typing import Optional, Tuple

class LinearRegression:
    """
    Linear Regression model implementation.
    
    Attributes:
        weights (np.ndarray): Model weights
        bias (float): Model bias term
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of training iterations
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000
    ):
        """
        Initialize Linear Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias