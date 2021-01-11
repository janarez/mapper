import numpy as np
from mapper_src.mapper import Mapper


points = np.load('spirals.npy')


mapper = Mapper(
    bins=7,
    overlap=0.04,
    clustering_function='tomato',
    distance=200,
    coordinate=0
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_graph_in_plane()
mapper.plot_persistence_homology()
