import numpy as np
from mapper import Mapper


with open('point_clouds/chair.txt') as f:
    data = f.readlines()

points = np.array([list(map(float, p.strip().split(' '))) for p in data])


mapper = Mapper(
    overlap=0.05,
    bins=8,
    coordinate=1
)

graph = mapper.fit(points)
mapper.plot_vertices()
mapper.plot_intervals()
mapper.plot_clusters()
mapper.plot_graph()
mapper.plot_graph_in_plane(seed=22)
mapper.plot_persistence_homology()
