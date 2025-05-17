"""
K-means clustering implementation from scratch.
"""

import numpy as np

np.random.seed(1337)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    """
    K-means model implementation.
    """
    def __init__(self, K=3, max_iter=100, plot_steps=False):
        self.K = K
        self.max_iter = max_iter
        self.plot_steps = plot_steps
        self.labels = None
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # mean feature vector for each cluster
        self.centroids = []
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        return np.argmin(distances)
    
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for i, sample in enumerate(self.X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(i)
        return clusters
    
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for i, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[i] = cluster_mean
        return centroids
    
    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_sample)
        for i, cluster in enumerate(clusters):
            for sample in cluster:
                labels[sample] = i
        return labels

    def predict(self, X):
        self.X = X
        self.n_sample, self.n_features = X.shape
        
        # initilize centroids
        random_samples = np.random.choice(self.n_sample, self.K, replace=False)
        self.centroids = [self.X[i] for i in random_samples]
        
        # optimization
        for _ in range(self.max_iter):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            #update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # check convergence
            if self._is_converged(centroids_old, self.centroids):
                break
        # return cluster labels
        return self._get_cluster_labels(self.clusters)