# Linear Regression

## Theory

Linear Regression is a fundamental supervised learning algorithm that models the relationship between a dependent variable (y) and one or more independent variables (X) using a linear equation.

### Mathematical Formulation

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

### Cost Function

The goal is to find the values of β that minimize the **Mean Squared Error (MSE)**:

```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

Where:
- `ŷᵢ` is the predicted value
- `yᵢ` is the actual value
- `n` is the number of samples

### Optimization Methods

1. **Gradient Descent**
   - Iteratively updates parameters in the direction of steepest descent
   - Requires learning rate (α) to control step size
   - Updates parameters using:
     ```
     β = β - α * ∇J(β)
     ```
   - Where ∇J(β) is the gradient of the cost function

2. **Normal Equation**
   - Direct solution: β = (X^T X)^(-1) X^T y
   - Computationally expensive for large datasets
   - No learning rate needed
   - Works well for small datasets

## When to Use

- **Use Linear Regression when**:
  - The relationship between variables is linear
  - You need a simple, interpretable model
  - You want to understand feature importance
  - You need fast predictions

- **Consider alternatives when**:
  - The relationship is non-linear
  - You have many features (consider regularization)
  - You need to capture complex patterns

## Advantages

1. Simple and interpretable
2. Computationally efficient
3. Works well with small datasets
4. Provides feature importance insights
5. Easy to implement and understand

## Limitations

1. Assumes linear relationship
2. Sensitive to outliers
3. Can't capture non-linear patterns
4. May underfit complex data
5. Requires feature scaling for better performance

## Applications

1. House price prediction
2. Sales forecasting
3. Risk assessment
4. Weather prediction
5. Economic modeling

## Best Practices

1. Always scale features
2. Handle outliers appropriately
3. Check for multicollinearity
4. Validate assumptions
5. Use cross-validation
6. Consider regularization for many features
