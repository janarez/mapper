import numpy as np

import sys
from mapper import Mapper


def noisy_circle(points=100, noise=0.1):
    """
    Generates points of a noisy circle (with radius 1).
    """

    # Generate circle coordinates.
    d = np.linspace(0, 2 * np.pi, points, endpoint=False)
    x = np.cos(d)
    y = np.sin(d)
    z = np.zeros(points)

    # Add Gaussian noise.
    x = np.random.normal(x, noise)
    y = np.random.normal(y, noise)

    return np.vstack((x, y, z)).T

points = noisy_circle()
print(points.shape)

mapper = Mapper(
    coordinate=1,
    bins=3,
    clustering_function="agglomerative",
    linkage="average",
    distance=1.5,
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_graph_in_plane()
mapper.plot_persistence_homology()
