# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import mapper as m

# %%
class HeightFilter(m.Filter):
    """
    Uses last coordinate of vertices.
    """

    def __call__(self, vertices):
        return vertices[:, -1]

class BinPartitioner(m.Partitioner):
    """
    Partitions into `n` bins with `p` percentage overlap.
    """

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self, numbers):
        # Get interval bounds.
        start = min(numbers)
        end = max(numbers)

        # Compute size and overlap of each bin.
        bin_size = (end - start) / self.n
        bin_overlap = self.p * bin_size

        # Compute intervals.
        bins = [[start + (bin_size - bin_overlap) * i,
                 start + bin_size * (i + 1)]
                for i in range(self.n)]
        return np.array(bins)

class SciKitClustering(m.Clustering):
    """
    Uses existing clustering implementation from `scikit-learn`.
    """

    def __init__(self, cluster):
        self.cluster = cluster

    def __call__(self, vertices):
        self.cluster.fit(vertices)
        return self.cluster.labels_

mapper = m.Mapper()
mapper.filter = HeightFilter()
mapper.partitioner = BinPartitioner(5, 0.25)
mapper.clustering = SciKitClustering(sklearn.cluster.AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.2
))

# %%
def noisy_circle(points=100, noise=0.1):
    """
    Generates points of a noisy circle (with radius 1).
    """

    # Generate circle coordinates.
    d = np.linspace(0, 2 * np.pi, points, endpoint=False)
    x = np.cos(d)
    y = np.sin(d)

    # Add Gaussian noise.
    x = np.random.normal(x, noise)
    y = np.random.normal(y, noise)

    return np.vstack((x, y)).T

vertices = noisy_circle()

# %%
mapper.process(vertices)
mapper.plot_vertices()

# %%
