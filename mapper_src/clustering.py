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
        return self._clustering.fit_predict(vertices)
