# K-Nearest Neighbors (KNN)

## Theory

K-Nearest Neighbors is a simple, non-parametric supervised learning algorithm used for both classification and regression. It makes predictions based on the k closest training examples in the feature space.

### How It Works

1. **Training Phase**
   - Store all training examples
   - No actual "training" occurs
   - Just memorizes the training data

2. **Prediction Phase**
   - For a new point:
     1. Find k nearest neighbors
     2. For classification: Take majority vote
     3. For regression: Take mean of neighbors

### Distance Metrics

Common distance measures:
1. **Euclidean Distance**
   ```
   d(x,y) = √(Σ(xᵢ - yᵢ)²)
   ```

2. **Manhattan Distance**
   ```
   d(x,y) = Σ|xᵢ - yᵢ|
   ```

3. **Minkowski Distance**
   ```
   d(x,y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
   ```

### Choosing K

- **Small k**:
  - More sensitive to noise
  - More complex decision boundaries
  - Higher variance

- **Large k**:
  - Smoother decision boundaries
  - Less sensitive to noise
  - Higher bias

## When to Use

- **Use KNN when**:
  - Data is small to medium-sized
  - Decision boundary is irregular
  - You need a simple, interpretable model
  - You want to avoid training time

- **Consider alternatives when**:
  - Dataset is very large
  - Features are high-dimensional
  - You need fast predictions
  - Memory is limited

## Advantages

1. Simple to understand and implement
2. No training phase
3. Naturally handles multi-class problems
4. Works well with non-linear data
5. Adapts easily to new training data

## Limitations

1. Computationally expensive for large datasets
2. Sensitive to irrelevant features
3. Requires feature scaling
4. Memory intensive
5. Sensitive to the choice of k

## Applications

1. Image classification
2. Pattern recognition
3. Recommendation systems
4. Anomaly detection
5. Medical diagnosis

## Best Practices

1. Always scale features
2. Choose appropriate distance metric
3. Use cross-validation to select k
4. Consider feature selection
5. Handle missing values appropriately
6. Use efficient data structures (e.g., KD-trees)

## Time and Space Complexity

- **Training**: O(1) time, O(n) space
- **Prediction**: O(n) time, O(1) space
- Where n is the number of training examples

## Variants

1. **Weighted KNN**
   - Weights neighbors by distance
   - Closer neighbors have more influence

2. **Radius-based KNN**
   - Uses fixed radius instead of k
   - Adapts to varying density

3. **KD-tree KNN**
   - Uses tree structure for faster search
   - Reduces prediction time complexity
