# K-means Clustering

## Theory

K-means is an unsupervised learning algorithm that partitions a dataset into $K$ distinct, non-overlapping clusters. It's one of the most popular clustering algorithms due to its simplicity and efficiency.

### How It Works

1. **Model Structure**
   - Each cluster is represented by a centroid (mean of points in the cluster)
   - Each data point is assigned to the nearest centroid

2. **Training Process**
   - Initialize $K$ centroids randomly
   - Assign each point to the nearest centroid
   - Update centroids as the mean of assigned points
   - Repeat assignment and update steps until convergence

### Mathematical Foundation

#### Objective Function
K-means minimizes the within-cluster sum of squares (WCSS):

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

Where:
- $k$ is the number of clusters
- $C_i$ is the set of points in cluster $i$
- $\mu_i$ is the centroid of cluster $i$
- $\|x - \mu_i\|^2$ is the squared Euclidean distance

## When to Use

- **Use K-means when**:
  - You want to partition data into $K$ groups
  - Data is continuous and clusters are roughly spherical
  - You need a fast, scalable algorithm

- **Consider alternatives when**:
  - Clusters are non-globular or vary in size/density
  - You don't know the optimal number of clusters
  - Data contains many outliers

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
4. Use elbow method to choose $K$
5. Handle outliers appropriately

## Time and Space Complexity

- **Training**: $O(n \cdot k \cdot i \cdot d)$
- **Space**: $O(n \cdot k)$
- Where:
  - $n$ = number of samples
  - $k$ = number of clusters
  - $i$ = number of iterations
  - $d$ = number of features

## Implementation Details

```python
from kmeans import KMeans

# Initialize and train
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.predict(X)
```

## Parameters

- `n_clusters` (int): Number of clusters
- `max_iter` (int): Maximum number of iterations
- `tol` (float): Tolerance for convergence
- `random_state` (int): Random seed for reproducibility

## Methods

- `fit(X)`: Train the model
- `predict(X)`: Assign clusters to new data
- `_initialize_centroids(X)`: Initialize cluster centers
- `_compute_distances(X)`: Calculate point-to-centroid distances
- `_update_centroids(X, labels)`: Update cluster centers

## License

This project is open source and available under the MIT License.
