# AdaBoost

## Theory

AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple weak learners to create a strong classifier. It works by sequentially training weak models, with each new model focusing more on the instances that previous models misclassified.

### How It Works

1. **Model Structure**
   - Ensemble of weak learners (decision stumps in this implementation)
   - Each weak learner is assigned a weight (alpha) based on its accuracy
   - Final prediction is a weighted majority vote

2. **Training Process**
   - Initialize equal weights for all training samples
   - For each iteration:
     - Train a weak learner to minimize weighted error
     - Calculate the learner's weight (alpha) based on its error
     - Update sample weights to focus more on misclassified instances
     - Add the weak learner to the ensemble

### Mathematical Foundation

#### Weak Learner Weight

The weight (alpha) of each weak learner is calculated as:

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$$

Where:
- $\alpha_t$ is the weight of the weak learner at iteration $t$
- $\epsilon_t$ is the weighted error rate of the weak learner

#### Sample Weight Update

After each iteration, the weights of the training samples are updated:

$$w_{i}^{(t+1)} = w_{i}^{(t)} \cdot \exp(-\alpha_t \cdot y_i \cdot h_t(x_i))$$

Where:
- $w_{i}^{(t)}$ is the weight of sample $i$ at iteration $t$
- $y_i$ is the true label of sample $i$
- $h_t(x_i)$ is the prediction of the weak learner at iteration $t$

#### Final Prediction

The final prediction is a weighted sum of all weak learners:

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(x)\right)$$

## When to Use

- **Use AdaBoost when**:
  - You have weak learners that perform slightly better than random
  - The dataset is not too noisy
  - You want a model that focuses on difficult examples
  - Interpretability and feature importance are desired

- **Consider alternatives when**:
  - Data is very noisy or contains many outliers
  - Computational efficiency is critical
  - You need a model that's less prone to overfitting
  - Base learners are already strong

## Advantages

1. Simple to implement
2. No need for prior knowledge about weak learner
3. Can identify hard-to-classify instances
4. Often resistant to overfitting
5. Provides feature importance measures

## Limitations

1. Sensitive to noisy data and outliers
2. Can overfit if weak learners are too complex
3. Sequential nature makes it hard to parallelize
4. Performance depends on weak learner selection
5. Training can be slow for large datasets

## Applications

1. Face detection
2. Text classification
3. Medical diagnosis
4. Fraud detection
5. Object recognition

## Best Practices

1. Use simple base learners (e.g., decision stumps)
2. Tune the number of weak learners
3. Ensure data quality and handle outliers
4. Consider early stopping to prevent overfitting
5. Scale features appropriately

## Time and Space Complexity

- **Training**: $O(T \cdot n \cdot d)$
- **Prediction**: $O(T \cdot d)$
- **Space**: $O(T)$

Where:
- $T$ = number of weak learners
- $n$ = number of samples
- $d$ = number of features

## Implementation Details

```python
from adaboost import AdaBoost

# Initialize model
model = AdaBoost(n_clf=5)  # Create ensemble with 5 weak learners

# Train model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## Parameters

- `n_clf` (int, default=5): Number of weak learners (decision stumps) to use

## Methods

- `fit(X, y)`: Train the AdaBoost model
  - Creates and trains weak learners sequentially
  - Updates sample weights at each iteration
  - Stores the ensemble of weak learners

- `predict(X)`: Make predictions
  - Combines predictions from all weak learners
  - Returns the sign of the weighted sum

## Helper Classes

- `DecisionStump`: A simple decision tree with only one split
  - `feature_i`: Feature index for the split
  - `threshold`: Threshold value for the split
  - `polar`: Direction of the split (1 or -1)
  - `alpha`: Weight assigned to this stump

## License

This project is open source and available under the MIT License.
