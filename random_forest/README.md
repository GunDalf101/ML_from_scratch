# Random Forest

## Theory

Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It uses bagging (bootstrap aggregating) and feature randomness to create a diverse set of trees whose predictions are aggregated through voting.

### How It Works

1. **Model Structure**
   - Ensemble of decision trees
   - Each tree is trained on a bootstrap sample of the data
   - Each tree uses a random subset of features at each split

2. **Training Process**
   - Create bootstrap samples from the training data
   - Train decision trees on these samples
   - At each node, consider only a random subset of features
   - Grow trees to their maximum depth or until stopping criteria are met

### Mathematical Foundation

#### Bootstrap Sampling

For each tree, a bootstrap sample is created by randomly sampling with replacement from the original dataset:

$$X_{boot}, y_{boot} = \text{sample with replacement}(X, y)$$

#### Feature Randomness

At each node, only a random subset of features is considered for splitting:

$$m = \sqrt{p}$$

Where:
- $m$ is the number of features to consider at each split
- $p$ is the total number of features

#### Prediction Aggregation

For classification, predictions are aggregated using majority voting:

$$\hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T)$$

Where:
- $\hat{y}_t$ is the prediction of the $t$-th tree
- $T$ is the total number of trees

## When to Use

- **Use Random Forest when**:
  - You need a highly accurate model
  - Data may have non-linear relationships
  - You want feature importance measures
  - You need robustness against overfitting

- **Consider alternatives when**:
  - Interpretability is critical
  - Training time is limited
  - Memory usage is a concern
  - You need a very simple model

## Advantages

1. High prediction accuracy
2. Robust to overfitting
3. Handles large datasets with high dimensionality
4. Provides feature importance measures
5. Works well with both categorical and numerical data

## Limitations

1. Less interpretable than single decision trees
2. Computationally intensive
3. Memory intensive for large forests
4. May overfit on noisy datasets
5. Slower prediction time than simpler models

## Applications

1. Classification and regression problems
2. Feature selection
3. Anomaly detection
4. Image classification
5. Bioinformatics

## Best Practices

1. Choose an appropriate number of trees
2. Consider the number of features at each split
3. Set appropriate tree depth
4. Balance accuracy and training time
5. Use cross-validation for hyperparameter tuning

## Time and Space Complexity

- **Training**: $O(T \cdot n \cdot \log(n) \cdot m)$
- **Prediction**: $O(T \cdot \log(n))$
- **Space**: $O(T \cdot n)$

Where:
- $T$ = number of trees
- $n$ = number of samples
- $m$ = number of features considered at each split

## Implementation Details

```python
from random_forest import RandomForestClassifier

# Initialize model
model = RandomForestClassifier(
    n_trees=100,              # Number of trees in the forest
    min_samples_split=2,      # Minimum samples required to split a node
    max_depth=100,            # Maximum depth of the trees
    n_features=None           # Number of features to consider at each split
)

# Train model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## Parameters

- `n_trees` (int, default=100): Number of trees in the forest
- `min_samples_split` (int, default=2): Minimum samples required to split a node
- `max_depth` (int, default=100): Maximum depth of the trees
- `n_features` (int, default=None): Number of features to consider at each split

## Methods

- `fit(X, y)`: Train the random forest model
  - Creates bootstrap samples
  - Trains individual decision trees
  - Stores trees in the ensemble

- `predict(X)`: Make predictions
  - Collects predictions from all trees
  - Returns majority vote for each sample

## Helper Functions

- `bootstrap_samples(X, y)`: Create bootstrap samples
- `most_common_label(y)`: Find the most common label in a set

## License

This project is open source and available under the MIT License.
