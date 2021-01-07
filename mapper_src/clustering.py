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
      
    def __call__(self, vertices, distance):
        self._clustering.set_params(**{self._distance_param : distance})
        labels = self._clustering.fit_predict(vertices)
        label_indices = [set() for _ in range(self._clustering.n_clusters_)]

        # Divide indices by labels.
        for i, l in enumerate(labels):
            label_indices[l].add(i)
            
        # Calculate center of each cluster.
        cluster_centers = []
        for indices in label_indices:
            convert_indices = operator.itemgetter(*indices)
            cluster_vertices = np.array(convert_indices(vertices))
            x, y = np.mean(cluster_vertices[:, 0]), np.mean(cluster_vertices[:, 1])
            cluster_centers.append((x, y))

        return labels, label_indices, cluster_centers