import numpy as np
from mapper_src.mapper import Mapper


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

points = noisy_circle()


mapper = Mapper(distance=2, linkage="average")

graph = mapper.fit(points)
mapper.plot_vertices()
mapper.plot_intervals()
mapper.plot_clusters()
mapper.plot_graph()
mapper.plot_persistence_homology()