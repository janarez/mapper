import operator
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN

class Clustering:
    """
    Function clustering vertices to nodes.
    Returns list of node indices for each vertex.
    """
    def __init__(self, clustering_function, **kwargs):
        if clustering_function == 'agglomerative':
            self._clustering = AgglomerativeClustering(
                n_clusters=None,
                compute_full_tree=True,
                linkage=kwargs.get('linkage', 'ward')
            )
            self._distance_param = 'distance_threshold'

        elif clustering_function == 'DBSCAN':
            self._clustering = DBSCAN()
            self._distance_param = 'eps'

        else:
            raise ValueError(f'Argument `clustering_function` must be one of: "agglomerative", "DBSCAN".')
      
    def __call__(self, vertices, indices, distance):
        self._clustering.set_params(**{self._distance_param : distance})
        labels = self._clustering.fit_predict(vertices)

        label_indices = [set() for _ in range(self._clustering.n_clusters_)]
        cluster_vertices = [list() for _ in range(self._clustering.n_clusters_)]
        # Divide indices by labels.
        for index, (i, l) in enumerate(zip(indices, labels)):
            label_indices[l].add(i)
            cluster_vertices[l].append(vertices[index])

        # Calculate center of each cluster.
        cluster_centers = [(np.mean(np.array(c)[:, 0]), np.mean(np.array(c)[:, 1])) for c in cluster_vertices]

        return labels, label_indices, cluster_centers