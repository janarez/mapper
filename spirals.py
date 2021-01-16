import numpy as np
from mapper import Mapper


points = np.load('point_clouds/spirals.npy')


mapper = Mapper(
    bins=7,
    overlap=0.04,
    coordinate=0
)

graph = mapper.fit(points)
mapper.plot_vertices()
mapper.plot_intervals()
mapper.plot_clusters()
mapper.plot_graph()
mapper.plot_graph_in_plane()
mapper.plot_persistence_homology()
