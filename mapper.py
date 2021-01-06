import numpy as np

class Mapper:
    def __init__(self):
        self.filter = Filter()
        self.partitioner = Partitioner()
        self.clustering = Clustering()

    def process(self, points):
        """
        Processes `points` and saves the results for further analysis.
        """

class Filter:
    """
    Function from points to some number (e.g., distance).
    """

    def __call__(self, points):
        return np.zeros(points.shape[0])

class Partitioner:
    """
    Function that partitions numbers to overlapping intervals.
    Returns list of intervals (each interval is list of numbers).
    """

    def __call__(self, numbers):
        return np.array([numbers]) # one interval

class Clustering:
    """
    Function clustering numbers to nodes.
    Returns list of nodes (each node is list of numbers).
    """

    def __call__(self, numbers):
        return np.array([numbers]) # one node
