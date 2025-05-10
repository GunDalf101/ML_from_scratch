# Logistic Regression from Scratch

This repository contains a simple implementation of Logistic Regression from scratch using NumPy. Logistic Regression is a classification algorithm used to predict binary outcomes (0 or 1) based on input features.

## Features

- Sigmoid activation function
- Gradient descent optimization
- Binary classification with customizable threshold
- Probability predictions
- Simple and intuitive API

## Installation

No additional installation is required beyond NumPy, which is used for numerical computations.

```bash
pip install numpy
```

## Usage

Here's a simple example of how to use the LogisticRegression class:

```python
from logistic_regression import LogisticRegression
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# Initialize and train the model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

## Parameters

- `learning_rate` (float, default=0.01): The step size used in gradient descent
- `n_iterations` (int, default=1000): Number of training iterations
- `threshold` (float, default=0.5): Decision boundary threshold for classification

## Methods

- `fit(X, y)`: Train the model using gradient descent
- `predict(X)`: Predict class labels (0 or 1)
- `predict_proba(X)`: Predict probability scores
- `decision_boundary(prob)`: Convert probability to class label

## Example

```python
# Create a more complex example
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Initialize model with custom parameters
model = LogisticRegression(learning_rate=0.01, n_iterations=2000, threshold=0.5)

# Train the model
model.fit(X, y)

# Get predictions and probabilities
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

## How It Works

1. The model uses the sigmoid function to convert linear predictions into probabilities
2. Gradient descent is used to optimize the weights and bias
3. The decision boundary (default 0.5) determines the final classification
4. The model can be used for binary classification tasks

## License

This project is open source and available under the MIT License.
