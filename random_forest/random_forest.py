"""
Random Forest implementation from scratch.
"""
import sys
import os

project_root = os.path.abspath("..") 
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from collections import Counter
from decision_tree.decision_tree_classifier import CustomDecisionTreeClassifier

def bootstrap_samples(X, y):
    n_sample = X.shape[0]
    idxs = np.random.choice(n_sample, size=n_sample, replace=True)
    return X[idxs], y[idxs]

def most_common_label(y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

class RandomForestClassifier:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_sample_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []    
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = CustomDecisionTreeClassifier(
                min_samples_split=self.min_sample_split,
                max_depth=self.max_depth,
                n_features=self.n_features)
            X_sample, y_sample = bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)