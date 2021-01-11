import numpy as np
from mapper_src.mapper import Mapper


points = np.load('lion.npy')


mapper = Mapper(
    bins=7,
    overlap=0.1,
    clustering_function='tomato',
    distance=6,
    filter_function='distance_from_point'
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_persistence_homology()
