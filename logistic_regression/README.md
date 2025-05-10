# Logistic Regression

## Theory

Logistic Regression is a statistical model used for binary classification that predicts the probability of an event occurring. Despite its name, it's a classification algorithm that uses the sigmoid function to transform linear predictions into probabilities.

### How It Works

1. **Model Structure**
   - Linear combination of features: z = wᵀx + b
   - Sigmoid activation: σ(z) = 1 / (1 + e^(-z))
   - Decision boundary: y = 1 if σ(z) ≥ 0.5, else y = 0

2. **Training Process**
   - Uses gradient descent to minimize binary cross-entropy loss
   - Updates weights and bias iteratively
   - Converges to optimal decision boundary

### Loss Function

Binary Cross-Entropy Loss:
```
L(y, ŷ) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

### Gradient Descent

Weight updates:
```
w = w - α * (1/m) * X^T * (ŷ - y)
b = b - α * (1/m) * sum(ŷ - y)
```

## When to Use

- **Use Logistic Regression when**:
  - You need a simple, interpretable model
  - Data is linearly separable
  - You want probability estimates
  - Training time is important
  - Memory usage should be minimal

- **Consider alternatives when**:
  - Data is highly non-linear
  - You need multi-class classification
  - Features are highly correlated
  - You need complex decision boundaries

## Advantages

1. Simple to understand and implement
2. Provides probability estimates
3. Fast training and prediction
4. Works well with high-dimensional data
5. Less prone to overfitting
6. Memory efficient

## Limitations

1. Assumes linear decision boundary
2. Sensitive to outliers
3. Requires feature scaling
4. May underfit complex data
5. Binary classification only (without modifications)

## Applications

1. Medical diagnosis
2. Credit scoring
3. Spam detection
4. Customer churn prediction
5. Fraud detection

## Best Practices

1. Always scale features
2. Handle class imbalance
3. Regularize to prevent overfitting
4. Use appropriate learning rate
5. Monitor convergence
6. Validate model assumptions

## Time and Space Complexity

- **Training**: O(n * d * i) time, O(d) space
- **Prediction**: O(d) time, O(1) space
- Where:
  - n = number of samples
  - d = number of features
  - i = number of iterations

## Implementation Details

```python
from logistic_regression import LogisticRegression
import numpy as np

# Initialize model
model = LogisticRegression(
    learning_rate=0.01,  # Step size for gradient descent
    n_iterations=1000,   # Number of training iterations
    threshold=0.5        # Decision boundary threshold
)

# Train model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

## Parameters

- `learning_rate` (float, default=0.01): Step size for gradient descent
- `n_iterations` (int, default=1000): Number of training iterations
- `threshold` (float, default=0.5): Decision boundary threshold

## Methods

- `fit(X, y)`: Train the model using gradient descent
- `predict(X)`: Predict class labels (0 or 1)
- `predict_proba(X)`: Predict probability scores
- `decision_boundary(prob)`: Convert probability to class label

## License

This project is open source and available under the MIT License.
