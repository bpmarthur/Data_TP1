"""
linear_scan module
"""
import numpy as np
from TD.nearest_neighbor import euclidean_distance as dist
from TD.nearest_neighbor import NearestNeighborSearch


class LinearScan(NearestNeighborSearch):
    def query(self, x):
        # Ensures x is of correct shape
        super().query(x)
        # Store the index of nearest neighbor
        nearest_neighbor_index = -1
        current_min_dist = np.inf
        # Ex2: Loop through points
        for index, el in enumerate(self.X):
            if current_min_dist > dist(el, x):
                current_min_dist = dist(el, x)
                nearest_neighbor_index = index
        return current_min_dist, nearest_neighbor_index
