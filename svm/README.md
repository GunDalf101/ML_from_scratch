# Support Vector Machine (SVM)

## Theory

Support Vector Machine is a powerful supervised learning algorithm used for classification and regression tasks. It finds the optimal hyperplane that maximizes the margin between classes, making it particularly effective for high-dimensional data.

### How It Works

1. **Model Structure**
   - Linear decision boundary: wᵀx + b = 0
   - Margin boundaries: wᵀx + b = ±1
   - Support vectors: Data points closest to the decision boundary

2. **Training Process**
   - Maximizes margin between classes
   - Uses kernel trick for non-linear classification
   - Solves constrained optimization problem

### Mathematical Foundation

#### Primal Form
The optimization problem in primal form:

$$
\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
$$

Subject to:
![Constraints](https://latex.codecogs.com/svg.image?\dpi{110}&space;y_i(w^T&space;x_i&space;+&space;b)&space;\geq&space;1&space;-&space;\xi_i,\quad&space;\xi_i&space;\geq&space;0)


Where:
- $w$ is the weight vector
- $b$ is the bias term
- $C$ is the regularization parameter
- $\xi_i$ are slack variables
- $y_i$ are class labels
- $x_i$ are feature vectors

#### Dual Form
The dual optimization problem:

$$
\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)
$$

Subject to:
![KKT Conditions](https://latex.codecogs.com/svg.image?\dpi{110}&space;0\leq\alpha_i\leq&space;C,\quad&space;\sum_{i=1}^n\alpha_i&space;y_i=0)


Where:
- $\alpha_i$ are Lagrange multipliers
- $K(x_i, x_j)$ is the kernel function

## When to Use

- **Use SVM when**:
  - Data is high-dimensional
  - Number of features is greater than number of samples
  - Clear margin of separation exists
  - You need a robust classifier

- **Consider alternatives when**:
  - Dataset is very large
  - Data is noisy
  - You need probability estimates
  - Training time is critical

## Advantages

1. Effective in high-dimensional spaces
2. Robust against overfitting
3. Versatile through kernel trick
4. Memory efficient (uses only support vectors)
5. Works well with clear margin of separation

## Limitations

1. Sensitive to noise
2. Computationally intensive for large datasets
3. Requires careful kernel selection
4. No direct probability estimates
5. Memory requirements scale with number of support vectors

## Applications

1. Text classification
2. Image classification
3. Bioinformatics
4. Face detection
5. Handwriting recognition

## Best Practices

1. Scale features before training
2. Choose appropriate kernel
3. Tune regularization parameter C
4. Handle class imbalance
5. Use cross-validation for parameter selection

## Time and Space Complexity

- **Training**: $O(n^2 \cdot d)$ to $O(n^3 \cdot d)$
- **Prediction**: $O(n_{sv} \cdot d)$
- **Space**: $O(n_{sv} \cdot d)$

Where:
- $n$ = number of samples
- $d$ = number of features
- $n_{sv}$ = number of support vectors

## Implementation Details

```python
from linear_svm import LinearSVM

# Initialize model
model = LinearSVM(
    learning_rate=0.001,  # Step size for gradient descent
    lambda_para=0.01,     # Regularization parameter
    n_iterations=1000     # Number of training iterations
)

# Train model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## Parameters

- `learning_rate` (float, default=0.001): Step size for gradient descent updates
- `lambda_para` (float, default=0.01): Regularization parameter that controls the trade-off between margin width and training error
- `n_iterations` (int, default=1000): Number of training iterations for gradient descent

## Methods

- `fit(X, y)`: Train the model using gradient descent
  - Converts labels to {-1, 1} format
  - Initializes weights and bias to zero
  - Updates parameters using gradient descent
  - Implements hinge loss with L2 regularization

- `predict(X)`: Make predictions
  - Returns sign of linear combination (wᵀx + b)
  - Outputs {-1, 1} for binary classification

## Helper Functions

- `linear(X, W, b)`: Computes linear combination wᵀx + b

## License

This project is open source and available under the MIT License.
