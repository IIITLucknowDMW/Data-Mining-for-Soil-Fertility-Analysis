import numpy as np
import random
from .metrics import distance


class K_MEANS:
    def __init__(self, k, distance_method, centroid_method, max_iterations):
        """
        K-Means Clustering Algorithm
        
        Parameters:
        - k: number of clusters
        - distance_method: 0 for Cosine, any other number for Minkowski with that order
        - centroid_method: 0 for random, 1 for k-means++
        - max_iterations: maximum number of iterations
        """
        self.k = k
        self.centroids = []
        self.dataset_labeled = None
        self.centroid_method = centroid_method
        self.distance_method = distance_method
        self.max_iterations = max_iterations
        self.X_train = None
    
    def fit(self, xt):
        """Fit the model to the data"""
        self.X_train = xt
        self.dataset_labeled = np.hstack((self.X_train.copy(), -1 * np.ones((self.X_train.shape[0], 1))))
    
    def centroid_selection(self, method):
        """Select initial centroids"""
        if method == 0:
            self.centroids.extend(self.X_train[random.sample(range(self.X_train.shape[0]), self.k), :])
        elif method == 1:
            self.centroids.append(list(self.X_train[np.random.choice(self.X_train.shape[0]), :]))
            dist = np.apply_along_axis(lambda x: distance(x, self.centroids[0], self.distance_method), axis=1, arr=self.X_train)
            ind = np.argsort(dist)
            for i in range(self.k, 0, -1):
                self.centroids.append(list(self.X_train[ind[int((len(ind) / self.k) * i) - 1], :]))
    
    def cluster(self):
        """Perform clustering"""
        self.centroid_selection(self.centroid_method)
        
        change = True
        num_iterations = 0
        
        while(change):
            for j in range(self.X_train.shape[0]):
                distances = []
                for i in range(self.k):
                    distances.append(distance(instance1=self.centroids[i], 
                                            instance2=self.X_train[j, :], 
                                            method=self.distance_method))
                cluster = np.argmin(distances)
                self.dataset_labeled[j, -1] = cluster
            
            old_centroids = self.centroids.copy()
            for i in range(self.k):
                cluster_points = np.array([row[:-1] for row in self.dataset_labeled if row[-1] == i])
                if len(cluster_points) > 0:
                    self.centroids[i] = np.array([np.average(cluster_points[:, j]) for j in range(cluster_points.shape[1])])

            if np.linalg.norm(np.array(self.centroids) - np.array(old_centroids)) < 0.0001 or num_iterations > self.max_iterations:
                change = False
            num_iterations += 1
        
        return self.dataset_labeled
    
    def predict(self, instance):
        """Predict cluster for new instance"""
        distances = []
        for i in range(self.k):
            distances.append(distance(self.centroids[i], instance, self.distance_method))
        cluster = np.argmin(distances)
        cluster_points = np.array([row[:-1] for row in self.dataset_labeled if row[-1] == cluster])
        return cluster, cluster_points
    
    def get_labels(self):
        """Get cluster labels"""
        if self.dataset_labeled is not None:
            return self.dataset_labeled[:, -1]
        return None


def silhouette_score_custom(data, labels, metric=2):
    """
    Calculate silhouette score for clustering
    
    Parameters:
    - data: feature data
    - labels: cluster labels
    - metric: distance method (0 for Cosine, other for Minkowski)
    
    Returns:
    - silhouette_score_avg: average silhouette score
    - intra_distance: total intra-cluster distance
    - inter_distance: total inter-cluster distance
    """
    num_points = len(data)
    unique_labels = np.unique(labels)
    silhouette_values = np.zeros(num_points)

    intra_cluster_distances = np.zeros(num_points)
    inter_cluster_distances = np.zeros(num_points)

    for i in range(num_points):
        label_i = labels[i]
        cluster_i_indices = np.where(labels == label_i)[0]
        
        if len(cluster_i_indices) == 1:
            silhouette_i = 0
        else:
            a_i = np.mean([distance(data[i], data[j], metric) for j in cluster_i_indices if j != i])
            intra_cluster_distances[i] = a_i

            b_i_values = []
            for label_j in unique_labels:
                if label_j != label_i:
                    cluster_j_indices = np.where(labels == label_j)[0]
                    b_ij = np.mean([distance(data[i], data[j], metric) for j in cluster_j_indices])
                    b_i_values.append(b_ij)
            
            b_i = min(b_i_values) if b_i_values else 0
            inter_cluster_distances[i] = b_i

            silhouette_i = (b_i - a_i) / max(a_i, b_i)
            
        silhouette_values[i] = silhouette_i
    
    silhouette_score_avg = np.mean(silhouette_values)
    intra_distance = np.sum(intra_cluster_distances)
    inter_distance = np.sum(inter_cluster_distances)

    return silhouette_score_avg, intra_distance, inter_distance