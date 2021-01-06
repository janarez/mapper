import numpy as np

class Clustering:
    """
    Function clustering vertices to nodes.
    Returns list of node indices for each vertex.
    """

    def __call__(self, vertices):
        return np.zeros(len(vertices)) # one node

class SciKitClustering(Clustering):
    """
    Uses existing clustering implementation from `scikit-learn`.
    """

    def __init__(self, cluster):
        self.cluster = cluster

    def __call__(self, vertices):
        self.cluster.fit(vertices)
        return self.cluster.labels_
