"""
Visualization utilities for machine learning algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model,
    title: str = "Decision Boundary",
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2"
) -> None:
    """
    Plot decision boundary for 2D classification problems.
    
    Args:
        X: Feature matrix (n_samples, 2)
        y: Target labels
        model: Trained classifier with predict method
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_learning_curve(
    train_scores: List[float],
    val_scores: List[float],
    title: str = "Learning Curve"
) -> None:
    """
    Plot learning curve showing training and validation scores.
    
    Args:
        train_scores: List of training scores
        val_scores: List of validation scores
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='Training Score')
    plt.plot(val_scores, label='Validation Score')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()
