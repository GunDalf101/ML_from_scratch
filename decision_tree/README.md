# Decision Tree Implementation

## Overview
A Decision Tree is a supervised learning algorithm that builds a tree-like model of decisions and their possible consequences. It's one of the most intuitive and interpretable machine learning algorithms, making it popular for both classification and regression tasks.

## Theory

### Basic Concepts
- **Node**: Each point in the tree where a decision is made
- **Root Node**: The topmost node representing the entire dataset
- **Internal Node**: Nodes that split into child nodes
- **Leaf Node**: Terminal nodes that make final predictions
- **Branch**: Path from root to leaf node

### Mathematical Foundation

#### Information Gain
The decision tree uses information gain to determine the best feature to split on:

\[
IG(D, A) = H(D) - H(D|A)
\]

Where:
- \(IG(D, A)\) is the information gain
- \(H(D)\) is the entropy of the dataset
- \(H(D|A)\) is the conditional entropy after splitting on feature A

#### Entropy
Entropy measures the impurity or disorder in a dataset:

\[
H(D) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

Where:
- \(p_i\) is the proportion of class i in the dataset
- \(c\) is the number of classes

#### Gini Impurity
An alternative to entropy, Gini impurity measures how often a randomly chosen element would be incorrectly labeled:

\[
Gini(D) = 1 - \sum_{i=1}^{c} p_i^2
\]

## Implementation Details

### Key Components
1. **Node Class**
   - Stores feature index and threshold for splitting
   - Maintains left and right child nodes
   - Stores prediction value for leaf nodes

2. **Tree Building**
   - Recursively splits data based on best feature and threshold
   - Uses information gain or Gini impurity for split selection
   - Implements early stopping criteria

3. **Prediction**
   - Traverses tree from root to leaf
   - Returns prediction based on leaf node value

### Methods
- `fit(X, y)`: Builds the decision tree
- `predict(X)`: Makes predictions for new data
- `_find_best_split(X, y)`: Finds optimal split point
- `_calculate_information_gain(X, y, feature_idx, threshold)`: Calculates information gain
- `_entropy(y)`: Calculates entropy of labels

## Advantages
1. Easy to understand and interpret
2. Can handle both numerical and categorical data
3. Requires little data preprocessing
4. Can capture non-linear relationships
5. Fast prediction time

## Limitations
1. Can overfit easily
2. Sensitive to small changes in data
3. Can create biased trees if classes are imbalanced
4. May not perform well with high-dimensional data

## Applications
1. Medical diagnosis
2. Credit risk assessment
3. Customer churn prediction
4. Fraud detection
5. Quality control

## Best Practices
1. Use pruning to prevent overfitting
2. Implement early stopping criteria
3. Consider feature importance
4. Handle missing values appropriately
5. Balance classes if necessary

## Complexity Analysis
- Time Complexity:
  - Training: O(n * m * log(n))
  - Prediction: O(log(n))
- Space Complexity: O(n)

Where:
- n = number of samples
- m = number of features

## Usage Example
```python
from decision_tree import DecisionTree

# Initialize and train
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)
```

## Dependencies
- NumPy
- scikit-learn (for demo and testing)
