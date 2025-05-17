# Perceptron

## Theory

The Perceptron is one of the simplest forms of artificial neural networks and serves as the building block for more complex neural networks. It's a binary classifier that learns a linear decision boundary through iterative updates.

### How It Works

1. **Model Structure**
   - Linear combination: z = wᵀx + b
   - Heaviside step activation: f(z) = 1 if z ≥ 0, else 0
   - Binary classification: y = f(z)

2. **Learning Process**
   - Initialize weights and bias to zero
   - For each training example:
     - Compute prediction: ŷ = f(wᵀx + b)
     - Update weights if prediction is wrong:
     ```
     w = w + α * (y - ŷ) * x
     b = b + α * (y - ŷ)
     ```
   where α is the learning rate

3. **Convergence**
   - Algorithm converges if data is linearly separable
   - Updates stop when no misclassifications occur
   - May not converge if data is not linearly separable

### Perceptron Learning Rule

The weight update rule follows:
```
Δw = α * (y - ŷ) * x
```
where:
- α is the learning rate
- y is the true label
- ŷ is the predicted label
- x is the input feature vector

## When to Use

- **Use Perceptron when**:
  - Data is linearly separable
  - You need a simple, interpretable model
  - Training time is important
  - Memory usage should be minimal
  - You want to understand basic neural networks

- **Consider alternatives when**:
  - Data is not linearly separable
  - You need probability estimates
  - You need multi-class classification
  - You need non-linear decision boundaries

## Advantages

1. Simple to understand and implement
2. Fast training and prediction
3. Memory efficient
4. Guaranteed convergence for linearly separable data
5. Serves as foundation for more complex neural networks
6. No hyperparameter tuning needed beyond learning rate

## Limitations

1. Only works for linearly separable data
2. No probability estimates
3. Binary classification only
4. Sensitive to feature scaling
5. May not converge if data is not linearly separable
6. No regularization mechanism

## Applications

1. Binary classification tasks
2. Educational purposes
3. Simple pattern recognition
4. Basic neural network understanding
5. Feature selection (through weight analysis)

## Best Practices

1. Scale features appropriately
2. Choose appropriate learning rate
3. Monitor convergence
4. Handle class imbalance
5. Consider data linearity
6. Use early stopping if needed

## Time and Space Complexity

- **Training**: O(n * d * i) time, O(d) space
- **Prediction**: O(d) time, O(1) space
- Where:
  - n = number of samples
  - d = number of features
  - i = number of iterations

## Implementation Details

```python
from perceptron import Perceptron
import numpy as np

# Initialize model
model = Perceptron(
    learning_rate=0.01,  # Step size for weight updates
    n_iterations=1000    # Maximum number of training iterations
)

# Train model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## Methods

- `fit(X, y)`: Train the model using the perceptron learning rule
- `predict(X)`: Predict class labels (0 or 1)
- `linear(X)`: Compute linear combination of features
- `Heaviside_step_fn(x)`: Apply Heaviside step activation function

## License

This project is open source and available under the MIT License.
