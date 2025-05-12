# Decision Tree

## Theory

Decision Tree is a supervised learning algorithm that builds a tree-like model of decisions and their possible consequences. It's popular for both classification and regression tasks due to its interpretability.

### How It Works

1. **Model Structure**
   - Nodes represent decisions or splits on features
   - Root node is the topmost node
   - Internal nodes split into child nodes
   - Leaf nodes make final predictions
   - Branches are paths from root to leaf

2. **Training Process**
   - Recursively split data based on the best feature and threshold
   - Use information gain or Gini impurity to select splits
   - Stop splitting based on early stopping criteria (e.g., max depth, min samples)

### Mathematical Foundation

#### Information Gain
The decision tree uses information gain to determine the best feature to split on:

$$
IG(D, A) = H(D) - H(D|A)
$$

Where:
- $IG(D, A)$ is the information gain
- $H(D)$ is the entropy of the dataset
- $H(D|A)$ is the conditional entropy after splitting on feature $A$

#### Entropy
Entropy measures the impurity or disorder in a dataset:

$$
H(D) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

Where:
- $p_i$ is the proportion of class $i$ in the dataset
- $c$ is the number of classes

#### Gini Impurity
An alternative to entropy, Gini impurity measures how often a randomly chosen element would be incorrectly labeled:

$$
Gini(D) = 1 - \sum_{i=1}^{c} p_i^2
$$

## When to Use

- **Use Decision Trees when**:
  - You need an interpretable model
  - Data has non-linear relationships
  - You want to handle both numerical and categorical data
  - Minimal data preprocessing is desired

- **Consider alternatives when**:
  - Data is high-dimensional
  - You want to avoid overfitting
  - Classes are highly imbalanced

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

## Time and Space Complexity

- **Training**: $O(n \cdot m \cdot \log(n))$
- **Prediction**: $O(\log(n))$
- **Space**: $O(n)$
- Where:
  - $n$ = number of samples
  - $m$ = number of features

## Implementation Details

```python
from decision_tree import DecisionTree

# Initialize and train
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)
```

## Parameters

- `max_depth` (int): Maximum depth of the tree
- `min_samples_split` (int): Minimum samples required to split a node
- `criterion` (str): Split criterion ('gini' or 'entropy')

## Methods

- `fit(X, y)`: Builds the decision tree
- `predict(X)`: Makes predictions for new data
- `_find_best_split(X, y)`: Finds optimal split point
- `_calculate_information_gain(X, y, feature_idx, threshold)`: Calculates information gain
- `_entropy(y)`: Calculates entropy of labels

## License

This project is open source and available under the MIT License.
