import numpy as np
from mapper import Mapper


with open('point_clouds/hand2.txt') as f:
    data = f.readlines()

points = np.array([list(map(float, p.strip().split(' '))) for p in data])


mapper = Mapper(
    clustering_function='tomato',
    distance=200,
    coordinate=1
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_graph_in_plane()
mapper.plot_persistence_homology()
