import numpy as np
from mapper_src.mapper import Mapper

def torus():
    u=np.linspace(0,2*np.pi,10)
    v=np.linspace(0,2*np.pi,50)
    u,v=np.meshgrid(u,v)
    a = 2
    b = 9
    x = (b + a*np.cos(u)) * np.cos(v)
    y = (b + a*np.cos(u)) * np.sin(v)
    z = a * np.sin(u)

    return np.vstack((x.ravel(), y.ravel(), z.ravel())).T

points = torus()


mapper = Mapper(
    bins=4,
    filter_function='by_coordinate',
    coordinate=1,
    clustering_function='agglomerative',
    linkage="average",
    distance=15
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_persistence_homology()

# Silhouette score
mapper = Mapper(
    bins=4,
    filter_function='by_coordinate',
    coordinate=1,
    clustering_function='agglomerative',
    linkage="average"
)

graph = mapper.fit(points)
mapper.plot_vertices_3d()
mapper.plot_intervals_3d()
mapper.plot_clusters_3d()
mapper.plot_graph_3d()
mapper.plot_persistence_homology()
