import numpy as np
from mapper_src.mapper import Mapper


with open('ant.txt') as f:
    data = f.readlines()

points = np.array([list(map(float, p.strip().split(' '))) for p in data])


mapper = Mapper(
    overlap=0.05,
    clustering_function='tomato',
    distance=20,
    coordinate=1,
    filter_function='distance_from_point',
    point=np.array([60, 80, 60])
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_persistence_homology()
