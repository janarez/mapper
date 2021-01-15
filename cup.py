import numpy as np
from mapper import Mapper


with open('point_clouds/cup.txt') as f:
    data = f.readlines()

points = np.array([list(map(float, p.strip().split(' '))) for p in data])


mapper = Mapper(
    bins=10,
    overlap=0.1,
    clustering_function='tomato',
    distance=50,
    coordinate=0
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_graph_in_plane(seed=22)
mapper.plot_persistence_homology()
