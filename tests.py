import numpy as np
from mapper_src.mapper import Mapper
from mapper_src.clusterer import SciKitClustering
from mapper_src.partitioner import BinPartitioner
import sklearn.cluster


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


mapper = Mapper()
mapper.partitioner = BinPartitioner(5, 0.25)
mapper.clustering = SciKitClustering(sklearn.cluster.AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=2
))

mapper.process(vertices)
mapper.plot_vertices()
mapper.plot_intervals()
mapper.plot_clusters()
