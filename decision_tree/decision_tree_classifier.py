"""
Decision Trees implementation from scratch.
"""
import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    """
    This is a single node of the tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        """Check if the node is a leaf."""
        return self.value is not None

class CustomDecisionTreeClassifier:
    """
    Decision Tree model implementation.
    
    Attributes:
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_sample_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _split(self, X_col, threshold):
        left_i = np.argwhere(X_col <= threshold).flatten()
        right_i = np.argwhere(X_col >= threshold).flatten()
        return left_i, right_i
    
    def _info_gain(self, y, X_col, threshold):
        # parent E
        parent_e = entropy(y)
        # generate split
        left_i, right_i = self._split(X_col, threshold)
        if len(left_i) == 0 or len (right_i) == 0:
            return 0
        # weighted avg child E
        n = len(y)
        n_l, n_r = len(left_i), len(right_i)
        e_l, e_r = entropy(y[left_i]), entropy(y[right_i])
        child_e = (n_l / n) * e_l + (n_r / n) * e_r
        
        # info gain
        info_gain = parent_e - child_e
        return info_gain
    
    def _best_criteria(self, X, y, feat_i):
        best_gain = -1
        split_i, split_thresh = None, None
        for i in feat_i:
            X_col = X[:, i]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._info_gain(y, X_col, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_i = i
                    split_thresh = threshold
        return split_i, split_thresh
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # stop
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_i = np.random.choice(n_features, self.n_features, replace=False)
        
        # greedy search
        best_feat, best_thresh = self._best_criteria(X, y, feat_i)
        left_is, right_is = self._split(X[:, best_feat], best_thresh)
        
        left = self._grow_tree(X[left_is, :], y[left_is], depth + 1)
        right = self._grow_tree(X[right_is, :], y[right_is], depth + 1)
        return Node(best_feat, best_thresh, left, right)
    
    def fit(self, X, y):
        # grow the tree
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        # traverse the tree
        return np.array([self._traverse_tree(x, self.root) for x in X])
