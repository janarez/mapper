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

mapper = Mapper(distance=1.5, bins=3, linkage="average", coordinate=1)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_persistence_homology()
