# Neural Network

## Theory

Neural Networks are computational models inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers that process information and learn patterns from data through a process of weight optimization.

### How It Works

1. **Model Structure**
   - Input layer: Receives the features
   - Hidden layers: Process information through weighted connections
   - Output layer: Produces the final prediction
   - Neurons use activation functions to introduce non-linearity

2. **Training Process**
   - Forward propagation: Input signals flow through the network
   - Cost calculation: Error between predictions and actual values is measured
   - Backpropagation: Error is propagated backwards to update weights
   - Weight updates using gradient descent or its variants

### Mathematical Foundation

#### Forward Propagation

For each layer $l$:

$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

Where:
- $Z^{[l]}$ is the weighted input to layer $l$
- $W^{[l]}$ is the weight matrix for layer $l$
- $A^{[l-1]}$ is the activation from the previous layer
- $b^{[l]}$ is the bias vector for layer $l$
- $g^{[l]}$ is the activation function for layer $l$

#### Cost Function

For binary classification:
$$J = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(a^{(i)}) + (1-y^{(i)}) \log(1-a^{(i)})]$$

For multi-class classification:
$$J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{K} y_j^{(i)} \log(a_j^{(i)})$$

#### Backpropagation

The partial derivatives of the cost function with respect to weights and biases:

$$\frac{\partial J}{\partial W^{[l]}} = \frac{1}{m} \delta^{[l]} \cdot (A^{[l-1]})^T$$
$$\frac{\partial J}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{[l]}$$

Where $\delta^{[l]}$ is the error term for layer $l$.

## When to Use

- **Use Neural Networks when**:
  - Data has complex patterns or non-linear relationships
  - Dataset is large
  - High accuracy is prioritized over interpretability
  - Problem involves image, speech, or text recognition
  - Traditional algorithms underperform

- **Consider alternatives when**:
  - Dataset is small
  - Interpretability is crucial
  - Computational resources are limited
  - Problem is simple and linear
  - Quick training time is important

## Advantages

1. Capable of learning complex non-linear relationships
2. Highly adaptable to various types of data
3. Automatic feature extraction in deep networks
4. Robust to noisy data when properly regularized
5. State-of-the-art performance on many tasks

## Limitations

1. Requires large amounts of data
2. Computationally intensive to train
3. Prone to overfitting without proper regularization
4. Difficult to interpret (black box)
5. Many hyperparameters to tune

## Applications

1. Image and speech recognition
2. Natural language processing
3. Time series prediction
4. Recommendation systems
5. Medical diagnosis

## Best Practices

1. Scale input features 
2. Use appropriate activation functions
3. Initialize weights properly
4. Apply regularization techniques
5. Implement mini-batch gradient descent
6. Monitor training and validation metrics

## Time and Space Complexity

- **Training**: $O(n \cdot m \cdot i \cdot \sum_{l} n_l \cdot n_{l-1})$
- **Prediction**: $O(m \cdot \sum_{l} n_l \cdot n_{l-1})$
- **Space**: $O(\sum_{l} n_l \cdot n_{l-1})$

Where:
- $n$ = number of samples
- $m$ = number of features
- $i$ = number of iterations
- $n_l$ = number of neurons in layer $l$

## Implementation Details

```python
from neural_network import NeuralNetwork

# Initialize model
model = NeuralNetwork(
    layer_sizes=[4, 8, 3],       # Input, hidden, output layers
    learning_rate=0.01,          # Learning rate for gradient descent
    n_iterations=1000,           # Number of training epochs
    activation='relu',           # Activation function for hidden layers
    batch_size=32,               # Mini-batch size
    lambda_param=0.01            # L2 regularization parameter
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Parameters

- `layer_sizes` (list): Number of neurons in each layer
- `learning_rate` (float, default=0.01): Step size for gradient descent
- `n_iterations` (int, default=1000): Number of training epochs
- `activation` (str, default='sigmoid'): Activation function ('sigmoid', 'relu', 'tanh')
- `batch_size` (int or None, default=None): Size of mini-batches
- `lambda_param` (float, default=0): L2 regularization parameter
- `random_state` (int or None, default=None): Random seed for reproducibility

## Methods

- `fit(X, y, verbose=False)`: Train the neural network
  - Preprocesses targets
  - Performs mini-batch gradient descent
  - Optionally displays training progress

- `predict(X)`: Predict class labels
  - Returns binary classifications or class indices

- `predict_proba(X)`: Predict probability estimates
  - Returns raw activation outputs

## License

This project is open source and available under the MIT License. 