import numpy as np
from mapper.mapper import Mapper


with open('point_clouds/pliers.txt') as f:
    data = f.readlines()

points = np.array([list(map(float, p.strip().split(' '))) for p in data])


mapper = Mapper(
    bins=5,
    clustering_function="agglomerative",
    linkage="average",
    coordinate=-1,
    max_k=5
)

graph = mapper.fit(points)
mapper.plot_vertices()
mapper.plot_intervals()
mapper.plot_clusters()
mapper.plot_graph()
mapper.plot_graph_in_plane()
mapper.plot_persistence_homology()
