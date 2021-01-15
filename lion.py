import numpy as np
from mapper import Mapper


points = np.load('point_clouds/lion.npy')


mapper = Mapper(
    bins=7,
    overlap=0.1,
    clustering_function='tomato',
    distance=6,
    filter_function='distance_from_point'
)

graph = mapper.fit(points)
mapper.plot_vertices()
mapper.plot_intervals()
mapper.plot_clusters()
mapper.plot_graph()
mapper.plot_persistence_homology()
