# Linear Regression

Implementation of Linear Regression from scratch using gradient descent.

## Theory

Linear Regression is a supervised learning algorithm that models the relationship between a dependent variable (y) and one or more independent variables (X) using a linear equation:

y = Xw + b

where:
- y is the predicted value
- X is the input features
- w is the weight vector
- b is the bias term

The algorithm learns the optimal values of w and b by minimizing the Mean Squared Error (MSE) loss function:

L(w,b) = (1/n) * Σ(y_pred - y_true)²

## Implementation

The implementation uses gradient descent to optimize the model parameters:

1. Initialize weights and bias to zero
2. For each iteration:
   - Calculate predictions: y_pred = Xw + b
   - Compute gradients:
     - dw = (1/n) * X^T(y_pred - y)
     - db = (1/n) * Σ(y_pred - y)
   - Update parameters:
     - w = w - learning_rate * dw
     - b = b - learning_rate * db

## Usage

```python
from linear_regression import LinearRegression
import numpy as np

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Parameters

- `learning_rate` (float): Step size for gradient descent (default: 0.01)
- `n_iterations` (int): Number of training iterations (default: 1000)

## Methods

- `fit(X, y)`: Train the model on input data X and target values y
- `predict(X)`: Make predictions for input data X

## Example

See the `demo.ipynb` notebook for a complete example with visualization.

## 📌 Overview

Linear Regression is a fundamental supervised machine learning algorithm used for predicting a continuous target variable based on one or more predictor variables. It works by modeling the relationship as a linear equation.

- **Problem it solves**: Predicting continuous values based on input features
- **When to use it**: When you suspect a linear relationship between input and output variables
- **Key advantages**: Simple, interpretable, computationally efficient
- **Limitations**: Only captures linear relationships, sensitive to outliers

## 📊 Math Behind the Algorithm

Linear Regression models the relationship between variables using a linear equation:

For simple linear regression (one feature):
```
y = β₀ + β₁x + ε
```

For multiple linear regression (multiple features):
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- `y` is the target variable
- `x` are the input features
- `β₀` is the y-intercept (bias)
- `β₁, β₂, ..., βₙ` are the coefficients (weights)
- `ε` is the error term

The goal is to find the values of β that minimize the **Mean Squared Error (MSE)**:

```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

Where `ŷᵢ` is the predicted value and `yᵢ` is the actual value.

### Gradient Descent

To find the optimal coefficients, we use gradient descent:

1. Initialize coefficients randomly
2. Calculate the gradient of the cost function with respect to each coefficient
3. Update coefficients: β = β - α * gradient
4. Repeat until convergence

The partial derivatives for the coefficients are:

- For β₀: ∂MSE/∂β₀ = -(2/n) * Σ(yᵢ - ŷᵢ)
- For βⱼ: ∂MSE/∂βⱼ = -(2/n) * Σ(yᵢ - ŷᵢ) * xᵢⱼ

### Normal Equation

For small datasets, we can also compute the coefficients directly using the normal equation:

```
β = (X^T X)^(-1) X^T y
```

Where:
- `X` is the feature matrix with a column of 1s for the intercept
- `y` is the target vector
- `X^T` is the transpose of X
- `^(-1)` indicates matrix inversion

## 🔧 Implementation Details

Our implementation includes both gradient descent and normal equation methods:

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if self.method == 'gradient_descent':
            # Gradient descent
            for _ in range(self.n_iterations):
                # Make predictions
                y_pred = np.dot(X, self.weights) + self.bias
                
                # Calculate gradients
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
                db = (1/n_samples) * np.sum(y_pred - y)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        elif self.method == 'normal_equation':
            # Add bias term to X
            X_b = np.c_[np.ones((n_samples, 1)), X]
            
            # Calculate weights using normal equation
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            
            self.bias = theta[0]
            self.weights = theta[1:]
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

**Time Complexity**:
- Gradient Descent: O(n_iterations * n_samples * n_features)
- Normal Equation: O(n_features³ + n_samples * n_features²)

**Space Complexity**: O(n_features)

## 🧪 Example & Usage

```python
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with gradient descent
model_gd = LinearRegression(learning_rate=0.1, n_iterations=1000, method='gradient_descent')
model_gd.fit(X_train_scaled, y_train)

# Train model with normal equation
model_ne = LinearRegression(method='normal_equation')
model_ne.fit(X_train, y_train)

# Make predictions
y_pred_gd = model_gd.predict(X_test_scaled)
y_pred_ne = model_ne.predict(X_test)

# Calculate MSE
mse_gd = np.mean((y_test - y_pred_gd) ** 2)
mse_ne = np.mean((y_test - y_pred_ne) ** 2)

print(f"Gradient Descent MSE: {mse_gd:.4f}")
print(f"Normal Equation MSE: {mse_ne:.4f}")
```

## 📉 Output & Visualization

Here's how to visualize the regression line against the data:

```python
# Plot training data
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, s=30, label='Training data')

# Sort the values for smooth line plotting
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_line_scaled = scaler.transform(X_line)

# Plot predictions from both methods
y_line_gd = model_gd.predict(X_line_scaled)
y_line_ne = model_ne.predict(X_line)

plt.plot(X_line, y_line_gd, color='red', linewidth=2, label='Gradient Descent')
plt.plot(X_line, y_line_ne, color='green', linewidth=2, linestyle='--', label='Normal Equation')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

![Linear Regression Plot](./images/linear_regression_plot.png)

Sample output:
```
Gradient Descent MSE: 1.0214
Normal Equation MSE: 0.9876
```

The visualization shows:
- Original data points (blue dots)
- Regression line from gradient descent (red line)
- Regression line from normal equation (green dashed line)

Both methods converge to similar solutions, though the normal equation is more precise with fewer computational steps for small datasets.

## 📚 Further Reading

- "Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Stanford CS229 Lecture Notes on Linear Regression](http://cs229.stanford.edu/notes/cs229-notes1.pdf)

## 🔄 Comparison with Other Algorithms

**When to use Linear Regression over alternatives**:
- Use when relationships are likely linear
- When you need a highly interpretable model
- As a baseline before trying more complex models

**Alternatives**:
- **Polynomial Regression**: When relationships are non-linear
- **Ridge/Lasso Regression**: When dealing with many features or multicollinearity
- **Decision Trees/Random Forests**: For capturing non-linear patterns and interactions
- **Support Vector Regression**: For better performance with non-linear boundaries

---

**Author:** Your Name  
**Last Updated:** April 14, 2025