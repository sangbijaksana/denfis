import numpy as np

class EvolvingClusteringMethod:
    """
    This class implements an evolving clustering method.

    Attributes:
        threshold_diameter (float): A threshold value used to determine the maximum diameter for a cluster.
        threshold_radius (float): Half of the threshold diameter, representing the maximum radius for a cluster.
        clusters (list): A dynamic list of clusters, each cluster containing a set of data points.
        cluster_centers (list): A list of the current centers for each cluster.
        cluster_radius (list): A list of the current radius for each cluster.
    """

    # Constructor for the EvolvingClusteringMethod class
    def __init__(self, threshold_diameter=0.5):

        self.threshold_diameter = threshold_diameter
        self.threshold_radius = threshold_diameter/2.0

        self.clusters = []
        self.cluster_centers = []
        self.cluster_radius = []

    # Function to update the clusters with a new data point
    def update_clusters(self, data_point):
        assigned = False
        for i, center in enumerate(self.cluster_centers):
            current_cluster_radius = self.cluster_radius[i]

            if np.linalg.norm(np.subtract(data_point, center)) <= current_cluster_radius:
                self.clusters[i].append(data_point)
                self.cluster_centers[i] = np.mean(self.clusters[i])
                assigned = True
                break

        for i, center in enumerate(self.cluster_centers):
            if assigned:
                break
            if np.linalg.norm(np.subtract(data_point, center)) <= self.threshold_radius:
                self.clusters[i].append(data_point)
                self.cluster_centers[i] = np.mean(self.clusters[i])
                self.cluster_radius[i] = np.linalg.norm(np.subtract(data_point, center))
                assigned = True
                break

        if not assigned:
            self.clusters.append([data_point])
            self.cluster_centers.append(data_point)
            self.cluster_radius.append(0)

        return not assigned