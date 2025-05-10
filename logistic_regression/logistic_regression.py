"""
Logistic regression implementation from scratch
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegression:
    """
    Logistic Regression Model implementation
    
    Attributes:
        weights (np.ndarray): Model weights
        bias (float): Model bias term
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of training iterations
        threshold (float): a value between 
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        threshold: float = 0.5
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #Gradient descent
        for _ in range(self.n_iterations):
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    
    def predict_proba(self, X):
        """
        Predicts the probabilities using the trained model.
        """
        return sigmoid((np.dot(X, self.weights) + self.bias))
    
    def decision_boundary(self, prob):
        """
        Decides the class based on the probability and the threshold.
        """
        return 1 if prob >= self.threshold else 0 
    
    def predict(self, X):
        """
        Predict Class using the trained model.
        """
        prediction = sigmoid((np.dot(X, self.weights) + self.bias))
        self.decision_boundary = np.vectorize(self.decision_boundary)
        return self.decision_boundary(prediction).flatten()
