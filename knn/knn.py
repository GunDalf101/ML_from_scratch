"""
K Nearest Neighbors implementation from scratch.
"""
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNearestNeighbors:
    """
    K Nearest Neighbors model implementation
    
    Attributes:
        k: number of nearest neighbors
    """
    
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        """
        Train the model by saving the training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x):
        # compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest labels
        k_i = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_i]
        # get most common class label
        majority = Counter(k_nearest_labels).most_common(1)
        return majority[0][0]
        
        