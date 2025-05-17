"""
Data preprocessing utilities for machine learning algorithms.
"""
import numpy as np
from typing import Tuple, Optional

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target values
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_samples = int(n_samples * test_size)
    
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalize features using min-max scaling.
    
    Args:
        X: Feature matrix
        
    Returns:
        Normalized feature matrix
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

def standardize(X: np.ndarray) -> np.ndarray:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Args:
        X: Feature matrix
        
    Returns:
        Standardized feature matrix
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std 