# Principal Component Analysis (PCA)

## Theory

Principal Component Analysis (PCA) is a dimensionality reduction technique that finds the directions (principal components) that maximize the variance in the data. It's widely used for feature extraction, data visualization, and preprocessing in machine learning pipelines.

### How It Works

1. **Model Structure**
   - Linear transformation that projects data onto orthogonal axes
   - Principal components are ordered by explained variance
   - Lower-dimensional representation preserves maximum variance

2. **Training Process**
   - Center the data by subtracting the mean
   - Compute the covariance matrix
   - Perform eigendecomposition of the covariance matrix
   - Select top k eigenvectors as principal components

### Mathematical Foundation

#### Covariance Matrix

The covariance matrix $\Sigma$ of the centered data matrix $X$:

$$\Sigma = \frac{1}{n-1}X^TX$$

Where:
- $X$ is the centered data matrix (each feature has zero mean)
- $n$ is the number of samples

#### Eigendecomposition

The eigendecomposition of the covariance matrix:

$$\Sigma v = \lambda v$$

Where:
- $v$ is an eigenvector (principal component)
- $\lambda$ is the corresponding eigenvalue (variance along that component)

#### Projection

The projection of data onto the principal components:

$$Z = X V$$

Where:
- $Z$ is the transformed data
- $V$ is the matrix of selected eigenvectors

## When to Use

- **Use PCA when**:
  - Dimensionality reduction is needed
  - You want to visualize high-dimensional data
  - Features are correlated
  - Computational efficiency is important
  - Noise reduction is desired

- **Consider alternatives when**:
  - Interpretability of features is critical
  - Non-linear relationships exist in the data
  - Supervised information should be preserved
  - Working with sparse data

## Advantages

1. Reduces dimensionality while preserving variance
2. Removes correlation between features
3. Reduces noise in the data
4. Improves computational efficiency
5. Helps with data visualization

## Limitations

1. Only captures linear relationships
2. Sensitive to feature scaling
3. May lose important information
4. Difficult to interpret principal components
5. Assumes orthogonal axes are optimal

## Applications

1. Image compression
2. Data visualization
3. Feature extraction
4. Noise reduction
5. Preprocessing for machine learning

## Best Practices

1. Scale features before applying PCA
2. Choose number of components based on explained variance
3. Consider domain knowledge when selecting components
4. Validate the information loss
5. Use incremental PCA for large datasets

## Time and Space Complexity

- **Training**: $O(min(nd^2, n^2d))$
- **Transformation**: $O(ndk)$
- **Space**: $O(d^2)$

Where:
- $n$ = number of samples
- $d$ = number of features
- $k$ = number of principal components

## Implementation Details

```python
from pca import PCA

# Initialize model
model = PCA(n_components=3)  # Reduce to 3 dimensions

# Fit the model
model.fit(X)

# Transform data
X_reduced = model.transform(X)
```

## Parameters

- `n_components` (int, default=3): Number of principal components to keep

## Methods

- `fit(X)`: Learn the principal components from the data
  - Centers the data by subtracting the mean
  - Computes covariance matrix
  - Performs eigendecomposition
  - Selects top components based on eigenvalues

- `transform(X)`: Project data onto principal components
  - Centers the data using the stored mean
  - Projects data onto the principal components

## License

This project is open source and available under the MIT License.
