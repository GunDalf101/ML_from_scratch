"""
Common evaluation metrics for machine learning algorithms.
"""
import numpy as np
from typing import Union, List

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        float: Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score for classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        np.ndarray: 2x2 confusion matrix
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])