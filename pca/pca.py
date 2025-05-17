"""
Principle Component Analysis implementation from scatch.
"""
import numpy as np

class PCA:
    """
    PCA model implementation.
    """
    def __init__(self, n_components=3): 
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        covar = np.cov(X.T)
        eigenvals, eigenvecs = np.linalg.eig(covar)
        eigenvecs = eigenvecs.T
        idxs = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idxs]
        eigenvecs = eigenvecs[idxs]
        self.components = eigenvecs[0:self.n_components]
    
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
    
