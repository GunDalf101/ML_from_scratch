# Gaussian Naive Bayes

## Theory

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of independence between features. It uses the normal (Gaussian) distribution to model the likelihood of features for each class.

### How It Works

1. **Bayes' Theorem**
   ```
   P(y|X) = P(X|y) * P(y) / P(X)
   ```
   where:
   - P(y|X) is the posterior probability
   - P(X|y) is the likelihood
   - P(y) is the prior probability
   - P(X) is the evidence

2. **Training Process**
   - Calculate class priors: P(y)
   - For each feature in each class:
     - Calculate mean (μ)
     - Calculate variance (σ²)
   - Store these parameters for prediction

3. **Prediction Process**
   - Calculate likelihood using Gaussian PDF:
   ```
   P(x|y) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
   ```
   - Multiply with class prior
   - Select class with highest posterior probability

### Naive Assumption

The "naive" in Naive Bayes comes from the assumption that all features are conditionally independent given the class:
```
P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)
```

## When to Use

- **Use Gaussian Naive Bayes when**:
  - Features are continuous and normally distributed
  - Dataset is small to medium-sized
  - You need fast training and prediction
  - You want probability estimates
  - Features are relatively independent

- **Consider alternatives when**:
  - Features are highly correlated
  - Data is not normally distributed
  - You need complex decision boundaries
  - Features are discrete/categorical

## Advantages

1. Fast training and prediction
2. Works well with high-dimensional data
3. Provides probability estimates
4. Handles missing data well
5. Less prone to overfitting
6. Memory efficient

## Limitations

1. Assumes feature independence
2. Assumes normal distribution
3. May perform poorly with correlated features
4. Sensitive to feature scaling
5. May be outperformed by more complex models

## Applications

1. Text classification
2. Spam detection
3. Medical diagnosis
4. Document categorization
5. Sentiment analysis

## Best Practices

1. Scale features appropriately
2. Handle missing values
3. Check feature independence
4. Verify normal distribution assumption
5. Use log probabilities for numerical stability
6. Consider feature selection

## Time and Space Complexity

- **Training**: O(n * d) time, O(c * d) space
- **Prediction**: O(c * d) time, O(1) space
- Where:
  - n = number of samples
  - d = number of features
  - c = number of classes

## Implementation Details

```python
from naive_bayes import GaussianNaiveBayes
import numpy as np

# Initialize model
model = GaussianNaiveBayes()

# Train model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

## Methods

- `fit(X, y)`: Train the model by calculating class priors and feature statistics
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `pdf(i, x)`: Calculate probability density for a feature in a class

## License

This project is open source and available under the MIT License.
