# K-means Clustering Implementation

## Overview
K-means is an unsupervised learning algorithm that partitions a dataset into K distinct, non-overlapping clusters. It's one of the most popular clustering algorithms due to its simplicity and efficiency.

## Theory

### Basic Concepts
- **Cluster**: A group of data points that are similar to each other
- **Centroid**: The center point of a cluster
- **K**: Number of clusters to form
- **Distance Metric**: Usually Euclidean distance

### Mathematical Foundation

#### Objective Function
K-means minimizes the within-cluster sum of squares (WCSS):

\[
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
\]

Where:
- \(k\) is the number of clusters
- \(C_i\) is the set of points in cluster i
- \(\mu_i\) is the centroid of cluster i
- \(\|x - \mu_i\|^2\) is the squared Euclidean distance

#### Algorithm Steps
1. Initialize K centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

## Implementation Details

### Key Components
1. **Centroid Initialization**
   - Random initialization
   - K-means++ initialization (optional)

2. **Cluster Assignment**
   - Compute distances to all centroids
   - Assign to nearest centroid

3. **Centroid Update**
   - Calculate mean of points in each cluster
   - Update centroid positions

### Methods
- `fit(X)`: Train the model
- `predict(X)`: Assign clusters to new data
- `_initialize_centroids(X)`: Initialize cluster centers
- `_compute_distances(X)`: Calculate point-to-centroid distances
- `_update_centroids(X, labels)`: Update cluster centers

## Advantages
1. Simple and easy to implement
2. Scales well to large datasets
3. Guarantees convergence
4. Works well with spherical clusters
5. Fast and efficient

## Limitations
1. Requires specifying number of clusters
2. Sensitive to initial centroid positions
3. Assumes spherical clusters
4. May converge to local optima
5. Not suitable for non-globular clusters

## Applications
1. Customer segmentation
2. Image compression
3. Document clustering
4. Anomaly detection
5. Data preprocessing

## Best Practices
1. Scale features before clustering
2. Use K-means++ initialization
3. Run multiple times with different seeds
4. Use elbow method to choose K
5. Handle outliers appropriately

## Complexity Analysis
- Time Complexity: O(n * k * i * d)
- Space Complexity: O(n * k)

Where:
- n = number of samples
- k = number of clusters
- i = number of iterations
- d = number of features

## Usage Example
```python
from kmeans import KMeans

# Initialize and train
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.predict(X)
```

## Dependencies
- NumPy
- scikit-learn (for demo and testing)
