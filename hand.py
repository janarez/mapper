import numpy as np
from mapper_src.mapper import Mapper


with open('hand.txt') as f:
    data = f.readlines()

points = np.array([list(map(float, p.strip().split(' '))) for p in data])


mapper = Mapper(distance=200, linkage="average")

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
# mapper.plot_clusters()
mapper.plot_graph()
mapper.plot_persistence_homology()
