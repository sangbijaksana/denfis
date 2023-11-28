import numpy as np

class MembershipFunction:
    """
    This class turn the cluster into membership function.

    Attributes:
        threshold_diameter (float): A threshold value used to determine the maximum diameter for a cluster.
        cluster_centers (float): Current centers cluster.
        width_of_triangle (float): Width of the triangle membership function.
    """
    def __init__(self, cluster_center, threshold_diameter, width_of_triangle=1.7):
        self.cluster_center = cluster_center
        self.threshold_diameter = threshold_diameter
        self.width_of_triangle = width_of_triangle

    def evaluate(self, input_values):
        x = input_values
        d = self.width_of_triangle

        b = self.cluster_center
        a = np.subtract(b, d*self.threshold_diameter)
        c = np.add(b, d*self.threshold_diameter)

        left_side_value = np.linalg.norm(x - a)/np.linalg.norm(b - a)
        right_side_value = np.linalg.norm(c - x)/np.linalg.norm(c - b)

        result = min(left_side_value, right_side_value)
        result = max(result, 0)
        return result
