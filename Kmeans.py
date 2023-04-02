from random import uniform
import numpy as np
from matplotlib import pyplot as plt


def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point is one-dimensional with a size (m,), data is two-dimensional with size (n,m),
    and returned value will be one-dimensional with size (n,).
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class KMeans:
    def __init__(self, n_clusters=2, max_iter=50):
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, data):
        # Select min and max values in data
        data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
        # initialize centroids with random points uniformly distributed in dataset domain (data_min, data_max)
        self.centroids = [uniform(data_min, data_max) for _ in range(self.n_clusters)]
        # initialize while loop conditionals
        iteration = 0
        prev_centroids = None
        # Iterate, adjusting centroids until converged or max_iter is exceeded
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Initialize empty list of list to store sorted points
            sorted_points = [[] for _ in range(self.n_clusters)]
            # Sort each datapoint, assigning to nearest centroid
            for x in data:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous
            prev_centroids = self.centroids
            # update current centroid with the mean of all points belonging to them
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1


if __name__ == '__main__':
    # Testing dataset given
    X = np.array([[2, 4],
                  [1.7, 2.8],
                  [7, 8],
                  [8.6, 8],
                  [3.4, 1.5],
                  [9, 11]])

    # Initialize Kmeans to create two separate clusters by default
    kmeans = KMeans()
    # Fit the two clusters against the X array
    kmeans.fit(X)

    # display the results over the initial dataset
    plt.scatter(X[:, 0], X[:, 1], s=150)
    plt.plot([x for x, _ in kmeans.centroids],
             [y for _, y in kmeans.centroids],
             '+',
             markersize=10,
             )
    plt.show()
